# CodeVerify

Verify AI-generated code in pull requests with a tree-sitter AST pass and a Claude review.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Analyze a diff from stdin
codeverify diff --repo owner/name --pr 42

# Post findings to a PR
codeverify post --repo owner/name --pr 42
```

## Architecture

1. **Diff parsing** (`diff_parser.py`) — extracts added lines from unified diffs
2. **AST analysis** (`ast_analyzer.py`) — tree-sitter scan for unresolved imports and undefined symbols
3. **LLM review** (`llm_reviewer.py`) — sends diff + AST hints to Anthropic Claude
4. **Validation** (`validator.py`) — filters findings by confidence (≥0.6) and line anchoring
5. **GitHub posting** (`github_poster.py`) — posts inline PR comments

## Environment

```
CODEVERIFY_GITHUB_TOKEN=ghp_xxx    # GitHub token
ANTHROPIC_API_KEY=sk-ant-xxx       # Anthropic API key
```

## License

MIT
