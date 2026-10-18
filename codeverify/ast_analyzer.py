"""Tree-sitter pass that flags unresolved imports and undefined symbols on added lines.

Scope and limits (by design, because the output is a *hint* for the LLM reviewer):
- Undefined-symbol checks need the complete post-change file. It is obtained through a
  ProjectContext and accepted only if its lines match the diff's added lines; otherwise only
  Python import resolution runs, using the diff's own lines.
- Symbols are resolved file-wide, ignoring scope. This under-reports rather than over-reports.
- Go and Rust resolve names per package/module, not per file, so their findings are labelled
  `possibly_undefined_*`.
- Python import resolution uses the environment codeverify runs in. Run it from the target
  project's virtualenv for accurate results.
"""

from __future__ import annotations

import builtins
import importlib.util
import logging
import re
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Protocol

from tree_sitter import Node, Parser
from tree_sitter_language_pack import get_parser

from codeverify.diff_parser import FileDiff
from codeverify.models import ASTFinding

logger = logging.getLogger(__name__)

MAX_AST_FINDINGS = 200
MAX_FILE_BYTES = 2_000_000
_MAX_LOCAL_MODULE_BASES = 16


class ProjectContext(Protocol):
    """Read access to the project at the revision being reviewed."""

    def read_file(self, path: str) -> str | None: ...

    def path_exists(self, path: str) -> bool: ...


class LocalProjectContext:
    """ProjectContext backed by a checkout on disk. Paths from the diff are untrusted."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _safe_path(self, relative: str) -> Path | None:
        candidate = (self._root / relative).resolve()
        if candidate == self._root or not candidate.is_relative_to(self._root):
            return None
        return candidate

    def read_file(self, path: str) -> str | None:
        target = self._safe_path(path)
        if target is None or not target.is_file():
            return None
        try:
            if target.stat().st_size > MAX_FILE_BYTES:
                return None
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            logger.debug("Cannot read %s: %s", target, exc)
            return None

    def path_exists(self, path: str) -> bool:
        target = self._safe_path(path)
        return target is not None and target.exists()


# --------------------------------------------------------------------------- facts


@dataclass(frozen=True)
class _Ref:
    symbol: str
    line: int


@dataclass(frozen=True)
class _Import:
    module: str
    line: int


@dataclass
class _Facts:
    imports: list[_Import] = field(default_factory=list)
    calls: list[_Ref] = field(default_factory=list)
    classes: list[_Ref] = field(default_factory=list)
    defined: set[str] = field(default_factory=set)
    wildcard: bool = False


@dataclass(frozen=True)
class _Language:
    group: str
    grammar: str
    extract: Callable[[Node], _Facts]
    builtins: frozenset[str]
    file_scoped: bool


def _text(node: Node | None) -> str:
    if node is None or node.text is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _line(node: Node) -> int:
    return node.start_point[0] + 1


def _walk(root: Node) -> Iterator[Node]:
    """Document-order traversal without recursion (deeply nested code is common in minified JS)."""
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def _identifiers(node: Node | None, kinds: frozenset[str]) -> Iterator[str]:
    if node is None:
        return
    for child in _walk(node):
        if child.type in kinds:
            text = _text(child)
            if text:
                yield text


def _words(blob: str) -> frozenset[str]:
    return frozenset(blob.split())


# --------------------------------------------------------------------------- Python

_PY_BUILTINS = frozenset(dir(builtins)) | _words(
    "__file__ __path__ __class__ __annotations__ __builtins__ __spec__ __loader__ "
    "__package__ __debug__"
)
_PY_PATTERN_CONTAINERS = frozenset(
    {
        "pattern_list",
        "tuple_pattern",
        "list_pattern",
        "list_splat_pattern",
        "dictionary_splat_pattern",
        "parenthesized_expression",
        "expression_list",
        "tuple",
        "list",
        "as_pattern_target",
    }
)
_PY_TRANSPARENT = frozenset({"parenthesized_expression", "await"})


def _py_target_names(node: Node | None) -> Iterator[str]:
    if node is None:
        return
    if node.type == "identifier":
        yield _text(node)
    elif node.type in _PY_PATTERN_CONTAINERS:
        for child in node.named_children:
            yield from _py_target_names(child)


def _py_root(node: Node | None) -> Node | None:
    """Leftmost identifier of an expression such as `a.b().c[0]`, or None."""
    while node is not None:
        kind = node.type
        if kind == "identifier":
            return node
        if kind == "attribute":
            node = node.child_by_field_name("object")
        elif kind == "call":
            node = node.child_by_field_name("function")
        elif kind == "subscript":
            node = node.child_by_field_name("value")
        elif kind in _PY_TRANSPARENT:
            node = node.named_children[0] if node.named_children else None
        else:
            return None
    return None


def _py_type_roots(type_node: Node) -> Iterator[Node]:
    for node in _walk(type_node):
        if node.type != "identifier":
            continue
        parent = node.parent
        if (
            parent is not None
            and parent.type == "attribute"
            and parent.child_by_field_name("object") != node
        ):
            continue
        yield node


def _py_params(params: Node | None, facts: _Facts) -> None:
    if params is None:
        return
    for child in params.named_children:
        if child.type in ("default_parameter", "typed_default_parameter"):
            facts.defined.update(_py_target_names(child.child_by_field_name("name")))
        elif child.type == "typed_parameter":
            first = child.named_children[0] if child.named_children else None
            facts.defined.update(_py_target_names(first))
        else:
            facts.defined.update(_py_target_names(child))


def _py_import(node: Node, facts: _Facts) -> None:
    line = _line(node)
    for child in node.children_by_field_name("name"):
        if child.type == "aliased_import":
            module = _text(child.child_by_field_name("name"))
            bound = _text(child.child_by_field_name("alias"))
        else:
            module = _text(child)
            bound = module.split(".")[0]
        if module:
            facts.imports.append(_Import(module, line))
        if bound:
            facts.defined.add(bound)


def _py_import_from(node: Node, facts: _Facts) -> None:
    facts.imports.append(_Import(_text(node.child_by_field_name("module_name")), _line(node)))
    for child in node.children_by_field_name("name"):
        if child.type == "aliased_import":
            bound = _text(child.child_by_field_name("alias"))
        else:
            bound = _text(child)
        if bound:
            facts.defined.add(bound)
    if any(child.type == "wildcard_import" for child in node.named_children):
        facts.wildcard = True


def _extract_python(root: Node) -> _Facts:
    facts = _Facts()
    for node in _walk(root):
        kind = node.type
        if kind == "import_statement":
            _py_import(node, facts)
        elif kind == "import_from_statement":
            _py_import_from(node, facts)
        elif kind == "function_definition":
            facts.defined.add(_text(node.child_by_field_name("name")))
            _py_params(node.child_by_field_name("parameters"), facts)
        elif kind == "lambda":
            _py_params(node.child_by_field_name("parameters"), facts)
        elif kind == "class_definition":
            facts.defined.add(_text(node.child_by_field_name("name")))
            supers = node.child_by_field_name("superclasses")
            if supers is not None:
                for child in supers.named_children:
                    ident = None if child.type == "keyword_argument" else _py_root(child)
                    if ident is not None:
                        facts.classes.append(_Ref(_text(ident), _line(ident)))
        elif kind in ("assignment", "augmented_assignment", "for_statement", "for_in_clause"):
            facts.defined.update(_py_target_names(node.child_by_field_name("left")))
        elif kind == "named_expression":
            facts.defined.update(_py_target_names(node.child_by_field_name("name")))
        elif kind == "as_pattern":
            facts.defined.update(_py_target_names(node.child_by_field_name("alias")))
        elif kind in ("global_statement", "nonlocal_statement"):
            facts.defined.update(_py_target_names_all(node))
        elif kind == "type_alias_statement":
            facts.defined.update(_identifiers(node.child_by_field_name("left"), _IDENT))
        elif kind in ("type_parameter", "case_pattern"):
            facts.defined.update(_identifiers(node, _IDENT))
        elif kind == "call":
            ident = _py_root(node.child_by_field_name("function"))
            if ident is not None:
                facts.calls.append(_Ref(_text(ident), _line(ident)))
        elif kind == "type":
            for ident in _py_type_roots(node):
                facts.classes.append(_Ref(_text(ident), _line(ident)))
    facts.defined.discard("")
    return facts


_IDENT = frozenset({"identifier"})


def _py_target_names_all(node: Node) -> Iterator[str]:
    for child in node.named_children:
        yield from _py_target_names(child)


# --------------------------------------------------------------------------- JS / TS

_JS_GLOBALS = _words(
    """
    undefined NaN Infinity globalThis arguments Object Function Array String Number Boolean Symbol
    BigInt Math Date RegExp Error EvalError RangeError ReferenceError SyntaxError TypeError URIError
    AggregateError JSON Promise Proxy Reflect Map Set WeakMap WeakSet WeakRef FinalizationRegistry
    ArrayBuffer SharedArrayBuffer DataView Atomics Int8Array Uint8Array Uint8ClampedArray Int16Array
    Uint16Array Int32Array Uint32Array Float32Array Float64Array BigInt64Array BigUint64Array Intl
    WebAssembly Iterator parseInt parseFloat isNaN isFinite eval decodeURI decodeURIComponent
    encodeURI encodeURIComponent escape unescape queueMicrotask structuredClone setTimeout
    clearTimeout setInterval clearInterval setImmediate clearImmediate console require module
    exports __dirname __filename process Buffer global fetch Headers Request Response FormData URL
    URLSearchParams AbortController AbortSignal TextEncoder TextDecoder Blob File FileReader atob
    btoa performance crypto window document navigator location history localStorage sessionStorage
    alert confirm prompt self top parent frames requestAnimationFrame cancelAnimationFrame
    requestIdleCallback XMLHttpRequest WebSocket Worker EventSource Event EventTarget CustomEvent
    MessageChannel BroadcastChannel Image Audio MutationObserver IntersectionObserver
    ResizeObserver HTMLElement Element Node Document DocumentFragment Text CSSStyleSheet
    getComputedStyle matchMedia addEventListener removeEventListener postMessage
    describe it test expect beforeEach afterEach beforeAll afterAll jest vi cy Cypress suite before
    after context specify xit xdescribe fdescribe fit assert define importScripts
    """
)
_TS_TYPES = _words(
    """
    Record Partial Required Readonly Pick Omit Exclude Extract NonNullable ReturnType Parameters
    ConstructorParameters InstanceType ThisType Awaited Uppercase Lowercase Capitalize Uncapitalize
    ReadonlyArray ReadonlyMap ReadonlySet PromiseLike ArrayLike Iterable IterableIterator
    AsyncIterable AsyncIterator Generator AsyncGenerator TemplateStringsArray PropertyKey
    """
)
_JS_NAME_KINDS = frozenset({"identifier", "type_identifier", "shorthand_property_identifier_pattern"})
_JS_VALUE_KINDS = frozenset({"identifier", "shorthand_property_identifier_pattern"})
_JS_DEFINITION_TYPES = frozenset(
    {
        "function_declaration",
        "generator_function_declaration",
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "enum_declaration",
        "internal_module",
        "module",
        "function_signature",
        "function_expression",
        "function",
        "generator_function",
        "class",
    }
)
_JS_TRANSPARENT = frozenset(
    {
        "parenthesized_expression",
        "non_null_expression",
        "await_expression",
        "as_expression",
        "satisfies_expression",
        "type_assertion",
    }
)


def _js_root(node: Node | None) -> Node | None:
    while node is not None:
        kind = node.type
        if kind == "identifier":
            return node
        if kind in ("member_expression", "subscript_expression"):
            node = node.child_by_field_name("object")
        elif kind == "call_expression":
            node = node.child_by_field_name("function")
        elif kind in _JS_TRANSPARENT:
            node = node.named_children[0] if node.named_children else None
        else:
            return None
    return None


def _extract_js(root: Node) -> _Facts:
    facts = _Facts()
    for node in _walk(root):
        kind = node.type
        if kind == "import_statement":
            module = _text(node.child_by_field_name("source")).strip("'\"`")
            facts.imports.append(_Import(module, _line(node)))
            facts.defined.update(_identifiers(node, _JS_NAME_KINDS))
        elif kind == "call_expression":
            func = node.child_by_field_name("function")
            if func is not None and func.type == "identifier" and _text(func) == "require":
                args = node.child_by_field_name("arguments")
                first = args.named_children[0] if args is not None and args.named_children else None
                if first is not None and first.type == "string":
                    facts.imports.append(_Import(_text(first).strip("'\"`"), _line(node)))
            ident = _js_root(func)
            if ident is not None:
                facts.calls.append(_Ref(_text(ident), _line(ident)))
        elif kind == "new_expression":
            ident = _js_root(node.child_by_field_name("constructor"))
            if ident is not None:
                facts.classes.append(_Ref(_text(ident), _line(ident)))
        elif kind == "class_heritage":
            for child in _walk(node):
                if child.type in ("identifier", "type_identifier"):
                    facts.classes.append(_Ref(_text(child), _line(child)))
        elif kind in _JS_DEFINITION_TYPES:
            facts.defined.update(_identifiers(node.child_by_field_name("name"), _JS_NAME_KINDS))
        elif kind == "variable_declarator":
            facts.defined.update(_identifiers(node.child_by_field_name("name"), _JS_VALUE_KINDS))
        elif kind == "formal_parameters":
            facts.defined.update(_identifiers(node, _JS_VALUE_KINDS))
        elif kind == "type_parameter":
            facts.defined.update(_identifiers(node, _JS_NAME_KINDS))
        elif kind == "arrow_function":
            facts.defined.update(
                _identifiers(node.child_by_field_name("parameter"), _JS_VALUE_KINDS)
            )
        elif kind == "catch_clause":
            facts.defined.update(
                _identifiers(node.child_by_field_name("parameter"), _JS_VALUE_KINDS)
            )
        elif kind == "for_in_statement":
            facts.defined.update(_identifiers(node.child_by_field_name("left"), _JS_VALUE_KINDS))
    return facts


# --------------------------------------------------------------------------- Go

_GO_PREDECLARED = _words(
    """
    len cap append copy delete make new panic recover print println close clear min max complex
    real imag true false nil iota error any comparable bool byte rune string int int8 int16 int32
    int64 uint uint8 uint16 uint32 uint64 uintptr float32 float64 complex64 complex128 _
    """
)
_GO_NAME_KINDS = frozenset({"identifier", "type_identifier", "package_identifier"})
_GO_NAME_FIELDS: dict[str, tuple[str, ...]] = {
    "function_declaration": ("name",),
    "method_declaration": ("name",),
    "type_spec": ("name",),
    "type_alias": ("name",),
    "var_spec": ("name",),
    "const_spec": ("name",),
    "short_var_declaration": ("left",),
    "range_clause": ("left",),
    "parameter_declaration": ("name",),
    "variadic_parameter_declaration": ("name",),
    "type_parameter_declaration": ("name",),
    "receive_statement": ("left",),
    "labeled_statement": ("label",),
    "type_switch_statement": ("alias",),
}


def _go_package_names(import_path: str) -> set[str]:
    """Plausible identifiers an import path binds (the package clause may differ from the path)."""
    parts = [part for part in import_path.split("/") if part]
    if not parts:
        return set()
    last = parts[-1]
    if re.fullmatch(r"v\d+", last) and len(parts) > 1:
        last = parts[-2]
    variants = {last, re.sub(r"\.v\d+$", "", last)}
    for variant in list(variants):
        variants.add(variant.removeprefix("go-").removesuffix("-go").removesuffix(".go"))
        variants.update(piece for piece in re.split(r"[-.]", variant) if piece)
    return {variant for variant in variants if variant.isidentifier()}


def _go_root(node: Node | None) -> Node | None:
    while node is not None:
        kind = node.type
        if kind == "identifier":
            return node
        if kind in ("selector_expression", "index_expression"):
            node = node.child_by_field_name("operand")
        elif kind == "call_expression":
            node = node.child_by_field_name("function")
        elif kind == "parenthesized_expression":
            node = node.named_children[0] if node.named_children else None
        else:
            return None
    return None


def _extract_go(root: Node) -> _Facts:
    facts = _Facts()
    for node in _walk(root):
        kind = node.type
        if kind == "import_spec":
            module = _text(node.child_by_field_name("path")).strip("\"`")
            facts.imports.append(_Import(module, _line(node)))
            alias = _text(node.child_by_field_name("name"))
            if alias == ".":
                facts.wildcard = True
            elif alias and alias != "_":
                facts.defined.add(alias)
            elif not alias:
                facts.defined.update(_go_package_names(module))
        elif kind in _GO_NAME_FIELDS:
            for field_name in _GO_NAME_FIELDS[kind]:
                for child in node.children_by_field_name(field_name):
                    facts.defined.update(_identifiers(child, _GO_NAME_KINDS))
        elif kind == "call_expression":
            ident = _go_root(node.child_by_field_name("function"))
            if ident is not None:
                facts.calls.append(_Ref(_text(ident), _line(ident)))
        elif kind == "composite_literal":
            type_node = node.child_by_field_name("type")
            if type_node is not None and type_node.type == "type_identifier":
                facts.classes.append(_Ref(_text(type_node), _line(type_node)))
            elif type_node is not None and type_node.type == "qualified_type":
                package = type_node.child_by_field_name("package")
                if package is not None:
                    facts.classes.append(_Ref(_text(package), _line(package)))
    return facts


# --------------------------------------------------------------------------- Rust

_RUST_PRELUDE = _words(
    """
    Some None Ok Err Vec String Box Option Result Self std core alloc proc_macro test drop Default
    Clone Copy ToString ToOwned Into From TryFrom TryInto AsRef AsMut Iterator IntoIterator Extend
    DoubleEndedIterator ExactSizeIterator Fn FnMut FnOnce Send Sync Sized Drop Eq PartialEq Ord
    PartialOrd FromIterator bool char str i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32
    f64
    """
)
_RUST_NAME_KINDS = frozenset(
    {"identifier", "type_identifier", "shorthand_field_identifier"}
)
_RUST_NAME_FIELDS: dict[str, tuple[str, ...]] = {
    "function_item": ("name",),
    "function_signature_item": ("name",),
    "struct_item": ("name",),
    "enum_item": ("name",),
    "union_item": ("name",),
    "trait_item": ("name",),
    "type_item": ("name",),
    "const_item": ("name",),
    "static_item": ("name",),
    "mod_item": ("name",),
    "macro_definition": ("name",),
    "associated_type": ("name",),
    "enum_variant": ("name",),
    "extern_crate_declaration": ("name", "alias"),
    "let_declaration": ("pattern",),
    "parameter": ("pattern",),
    "for_expression": ("pattern",),
    "let_condition": ("pattern",),
    "if_let_expression": ("pattern",),
    "while_let_expression": ("pattern",),
    "match_arm": ("pattern",),
    "type_parameter": ("name",),
    "const_parameter": ("name",),
    "constrained_type_parameter": ("left",),
}
_RUST_TRANSPARENT = frozenset(
    {
        "index_expression",
        "try_expression",
        "await_expression",
        "parenthesized_expression",
        "reference_expression",
        "unary_expression",
    }
)


def _join_path(prefix: str, name: str) -> str:
    return f"{prefix}::{name}" if prefix and name else prefix or name


def _rust_bind(facts: _Facts, path: str, bound: str, line: int) -> None:
    if path:
        facts.imports.append(_Import(path, line))
    if bound:
        facts.defined.add(bound)


def _rust_use(node: Node | None, prefix: str, line: int, facts: _Facts) -> None:
    if node is None:
        return
    kind = node.type
    if kind == "identifier":
        _rust_bind(facts, _join_path(prefix, _text(node)), _text(node), line)
    elif kind == "self":
        _rust_bind(facts, prefix, prefix.rsplit("::", 1)[-1], line)
    elif kind == "scoped_identifier":
        name = _text(node.child_by_field_name("name"))
        _rust_bind(facts, _join_path(prefix, _text(node)), name, line)
    elif kind == "use_as_clause":
        path = _join_path(prefix, _text(node.child_by_field_name("path")))
        _rust_bind(facts, path, _text(node.child_by_field_name("alias")), line)
    elif kind == "scoped_use_list":
        new_prefix = _join_path(prefix, _text(node.child_by_field_name("path")))
        _rust_use(node.child_by_field_name("list"), new_prefix, line, facts)
    elif kind == "use_list":
        for child in node.named_children:
            _rust_use(child, prefix, line, facts)
    elif kind == "use_wildcard":
        facts.wildcard = True


def _rust_root(node: Node | None) -> Node | None:
    while node is not None:
        kind = node.type
        if kind in ("identifier", "type_identifier"):
            return node
        if kind == "scoped_identifier":
            node = node.child_by_field_name("path")
        elif kind == "field_expression":
            node = node.child_by_field_name("value")
        elif kind in ("generic_function", "call_expression"):
            node = node.child_by_field_name("function")
        elif kind in ("generic_type", "generic_type_with_turbofish"):
            node = node.child_by_field_name("type")
        elif kind in _RUST_TRANSPARENT:
            node = node.named_children[0] if node.named_children else None
        else:
            return None
    return None


def _extract_rust(root: Node) -> _Facts:
    facts = _Facts()
    for node in _walk(root):
        kind = node.type
        if kind == "use_declaration":
            _rust_use(node.child_by_field_name("argument"), "", _line(node), facts)
        elif kind in _RUST_NAME_FIELDS:
            for field_name in _RUST_NAME_FIELDS[kind]:
                for child in node.children_by_field_name(field_name):
                    facts.defined.update(_identifiers(child, _RUST_NAME_KINDS))
        elif kind == "closure_parameters":
            facts.defined.update(_identifiers(node, _RUST_NAME_KINDS))
        elif kind == "call_expression":
            ident = _rust_root(node.child_by_field_name("function"))
            if ident is not None:
                facts.calls.append(_Ref(_text(ident), _line(ident)))
        elif kind == "struct_expression":
            ident = _rust_root(node.child_by_field_name("name"))
            if ident is not None:
                facts.classes.append(_Ref(_text(ident), _line(ident)))
    return facts


# --------------------------------------------------------------------------- registry

_LANGUAGES: dict[str, _Language] = {
    "python": _Language("python", "python", _extract_python, _PY_BUILTINS, True),
    "javascript": _Language("javascript", "javascript", _extract_js, _JS_GLOBALS, True),
    "typescript": _Language("typescript", "typescript", _extract_js, _JS_GLOBALS | _TS_TYPES, True),
    "tsx": _Language("typescript", "tsx", _extract_js, _JS_GLOBALS | _TS_TYPES, True),
    "go": _Language("go", "go", _extract_go, _GO_PREDECLARED, False),
    "rust": _Language("rust", "rust", _extract_rust, _RUST_PRELUDE, False),
}
_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
}


@lru_cache(maxsize=None)
def _parser(grammar: str) -> Parser:
    return get_parser(grammar)


def _language_for(path: str, enabled: frozenset[str]) -> _Language | None:
    grammar = _EXTENSIONS.get(PurePosixPath(path).suffix.lower())
    if grammar is None:
        return None
    language = _LANGUAGES[grammar]
    return language if language.group in enabled else None


# --------------------------------------------------------------------------- analysis


def analyze_files(
    files: Sequence[FileDiff], languages: Sequence[str], context: ProjectContext
) -> list[ASTFinding]:
    """Run the AST pass over every supported changed file. Never raises for a single bad file."""
    enabled = frozenset(language.lower() for language in languages)
    unique: dict[tuple[str, int, str, str], ASTFinding] = {}
    for file_diff in files:
        language = _language_for(file_diff.path, enabled)
        if language is None or not file_diff.added:
            continue
        try:
            findings = _analyze_file(file_diff, language, context)
        except Exception:  # noqa: BLE001 - the AST pass is advisory; one bad file must not stop a review
            logger.warning("AST analysis failed for %s", file_diff.path, exc_info=True)
            continue
        for finding in findings:
            unique.setdefault((finding.file, finding.line, finding.kind, finding.symbol), finding)

    ordered = sorted(unique.values(), key=lambda f: (f.file, f.line, f.kind, f.symbol))
    if len(ordered) > MAX_AST_FINDINGS:
        logger.warning("Truncating AST findings from %d to %d", len(ordered), MAX_AST_FINDINGS)
        ordered = ordered[:MAX_AST_FINDINGS]
    return ordered


def _analyze_file(
    file_diff: FileDiff, language: _Language, context: ProjectContext
) -> list[ASTFinding]:
    path = file_diff.path
    content = context.read_file(path)
    has_full_file = content is not None and _matches_diff(content, file_diff)
    if has_full_file and content is not None:
        source = content
    elif language.grammar == "python":
        logger.info("Full file unavailable for %s; checking imports only", path)
        source = _reconstruct(file_diff)
    else:
        logger.info("Full file unavailable for %s; skipping AST pass", path)
        return []

    root = _parser(language.grammar).parse(source.encode("utf-8")).root_node
    facts = language.extract(root)
    complete = has_full_file and not root.has_error
    if has_full_file and root.has_error:
        logger.info("Parse errors in %s; skipping undefined-symbol checks", path)

    added = file_diff.added
    findings: list[ASTFinding] = []

    if language.group == "python":
        for imp in facts.imports:
            if imp.line in added and not _python_import_resolves(imp.module, path, context):
                findings.append(
                    ASTFinding(file=path, line=imp.line, symbol=imp.module, kind="unresolved_import")
                )

    if complete and not facts.wildcard:
        known = facts.defined | language.builtins
        prefix = "" if language.file_scoped else "possibly_"
        for ref in facts.calls:
            if ref.line in added and ref.symbol not in known:
                findings.append(
                    ASTFinding(
                        file=path, line=ref.line, symbol=ref.symbol, kind=f"{prefix}undefined_call"
                    )
                )
        for ref in facts.classes:
            if ref.line in added and ref.symbol not in known:
                findings.append(
                    ASTFinding(
                        file=path, line=ref.line, symbol=ref.symbol, kind=f"{prefix}undefined_class"
                    )
                )
    return findings


def _matches_diff(content: str, file_diff: FileDiff) -> bool:
    """True if the file's lines at the added line numbers equal the diff's added lines."""
    lines = content.split("\n")
    for number, text in file_diff.added.items():
        if number > len(lines) or lines[number - 1].rstrip("\r") != text.rstrip("\r"):
            return False
    return bool(file_diff.added)


def _reconstruct(file_diff: FileDiff) -> str:
    """Build source text from diff lines only, keeping real line numbers (gaps become blanks)."""
    lines = file_diff.new_side()
    last = max(lines)
    return "\n".join(lines.get(number, "") for number in range(1, last + 1))


def _python_import_resolves(module: str, file_path: str, context: ProjectContext) -> bool:
    if module.startswith("."):
        return True
    top = module.split(".")[0]
    if not top or top in sys.stdlib_module_names:
        return True
    try:
        if importlib.util.find_spec(top) is not None:
            return True
    except ValueError:
        return True
    except ImportError:
        pass
    return _local_module_exists(top, file_path, context)


def _local_module_exists(top: str, file_path: str, context: ProjectContext) -> bool:
    parts = PurePosixPath(file_path).parent.parts
    bases = {"/".join(parts[:count]) for count in range(len(parts) + 1)}
    bases |= {f"{base}/src" if base else "src" for base in bases}
    for base in sorted(bases, key=len)[:_MAX_LOCAL_MODULE_BASES]:
        prefix = f"{base}/" if base else ""
        if context.path_exists(f"{prefix}{top}") or context.path_exists(f"{prefix}{top}.py"):
            return True
    return False
