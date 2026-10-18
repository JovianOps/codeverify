"""Command-line interface."""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import click

from codeverify import __version__
from codeverify.ast_analyzer import LocalProjectContext
from codeverify.config import Config, load_config
from codeverify.errors import CodeVerifyError, ConfigError
from codeverify.github_poster import (
    GitHubProjectContext,
    build_review_payload,
    fetch_pull_request,
    open_repository,
    post_review,
)
from codeverify.main import load_diff, load_result, run_review
from codeverify.models import ReviewResult

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


@contextmanager
def _cli_errors() -> Iterator[None]:
    try:
        yield
    except CodeVerifyError as exc:
        raise click.ClickException(str(exc)) from exc


def _validate_repo(_ctx: click.Context, _param: click.Parameter, value: str) -> str:
    if not _REPO_RE.fullmatch(value):
        raise click.BadParameter("must look like owner/name")
    return value


def _config(ctx: click.Context) -> Config:
    path = ctx.obj["config_path"]
    try:
        return load_config(path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc


def _emit(result: ReviewResult, output: Path | None) -> None:
    text = result.model_dump_json(indent=2)
    if output is None:
        click.echo(text)
        return
    try:
        output.write_text(text + "\n", encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"Cannot write {output}: {exc}") from exc
    click.echo(f"Wrote {len(result.findings)} finding(s) to {output}", err=True)


def _print_payload(result: ReviewResult) -> None:
    click.echo(json.dumps(dataclasses.asdict(build_review_payload(result)), indent=2))


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="codeverify")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Path to config.yaml (default: ./config.yaml if present; env vars are the fallback).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging on stderr.")
@click.pass_context
def cli(ctx: click.Context, config_path: Path | None, verbose: bool) -> None:
    """Verify AI-generated code in pull requests."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    ctx.obj = {"config_path": config_path}


@cli.command("diff")
@click.argument("source")
@click.option("-d", "--description", default="", help="PR description to give the reviewer.")
@click.option(
    "--repo-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=None,
    help="Checkout used to read full files for the AST pass (default: current directory).",
)
@click.option("-o", "--output", type=click.Path(path_type=Path, dir_okay=False), default=None)
@click.pass_context
def diff_command(
    ctx: click.Context, source: str, description: str, repo_root: Path | None, output: Path | None
) -> None:
    """Analyze a diff file ('-' for stdin) or a git ref and print the result as JSON.

    SOURCE is a unified-diff file, or a git ref: `A..B` / `A...B` diffs a range, a single
    ref diffs that commit against its parent.
    """
    config = _config(ctx)
    root = repo_root if repo_root is not None else Path.cwd()
    with _cli_errors():
        diff_text = load_diff(source, root)
        result = run_review(diff_text, description, config, LocalProjectContext(root))
    _emit(result, output)


@cli.command("pr")
@click.argument("number", type=click.IntRange(min=1))
@click.option("--repo", required=True, callback=_validate_repo, help="owner/name")
@click.option("--dry-run", is_flag=True, help="Print the review payload instead of posting it.")
@click.pass_context
def pr_command(ctx: click.Context, number: int, repo: str, dry_run: bool) -> None:
    """Fetch a pull request's diff, analyze it, and post the review (needs GITHUB_TOKEN)."""
    config = _config(ctx)
    with _cli_errors():
        repository = open_repository(repo)
        pull = fetch_pull_request(repository, number)
        context = GitHubProjectContext(repository, pull.head_sha)
        result = run_review(pull.diff, pull.description, config, context)
        if dry_run:
            _print_payload(result)
            return
        url = post_review(repository, number, result, commit_sha=pull.head_sha)
    click.echo(f"Posted review with {len(result.findings)} finding(s): {url}")


@cli.command("post")
@click.option(
    "--result",
    "result_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="JSON file produced by `codeverify diff`.",
)
@click.option("--pr", "number", required=True, type=click.IntRange(min=1))
@click.option("--repo", required=True, callback=_validate_repo, help="owner/name")
@click.option("--dry-run", is_flag=True, help="Print the review payload instead of posting it.")
def post_command(result_path: Path, number: int, repo: str, dry_run: bool) -> None:
    """Post a pre-computed result as a PR review (needs GITHUB_TOKEN)."""
    with _cli_errors():
        result = load_result(result_path)
        if dry_run:
            _print_payload(result)
            return
        url = post_review(open_repository(repo), number, result)
    click.echo(f"Posted review with {len(result.findings)} finding(s): {url}")


if __name__ == "__main__":
    cli()
