# CodeVerify

Verify AI-generated code in pull requests with a tree-sitter AST pass and a Claude review.

## Install

```bash
pip install git+https://github.com/JovianOps/codeverify.git
```

## Usage

```bash
# Analyze a PR diff
codeverify diff --repo owner/name --pr 42

# Fetch a PR, analyze it, and post review
codeverify pr --repo owner/name --pr 42
```

## Commands

- `diff` — Analyze a diff file (`-` for stdin) or git ref
- `pr` — Fetch a pull request's diff, analyze it, and post findings
- `post` — Post a pre-computed result as a PR review (needs `GITHUB_TOKEN`)

## Architecture

1. **Diff parsing** (`diff_parser.py`) — extracts added lines from unified diffs
2. **AST analysis** (`ast_analyzer.py`) — tree-sitter scan for unresolved imports and undefined symbols
3. **LLM review** (`llm_reviewer.py`) — sends diff + AST hints to Anthropic Claude
4. **Validation** (`validator.py`) — filters findings by confidence (≥0.6) and line anchoring
5. **GitHub posting** (`github_poster.py`) — posts inline PR comments

## Environment

```
CODEVERIFY_GITHUB_TOKEN=ghp_xxx    # GitHub token (for PR posting)
ANTHROPIC_API_KEY=sk-ant-xxx       # Anthropic API key
```

## License

MIT