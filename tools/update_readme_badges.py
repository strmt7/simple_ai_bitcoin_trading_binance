"""Generate the README badge block from canonical repository metadata."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlparse

BADGE_BLOCK_BEGIN = "<!-- BEGIN GENERATED BADGES -->"
BADGE_BLOCK_END = "<!-- END GENERATED BADGES -->"
README_PATH = Path("README.md")
CANONICAL_METADATA_PATH = Path(".github/readme_badges.json")


@dataclass(frozen=True)
class RepoMetadata:
    """Repository details needed to render badge URLs."""

    host: str
    owner: str
    repo: str
    remote_name: str
    branch_name: str

    @property
    def github_path(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def branch_query_value(self) -> str:
        return quote(self.branch_name, safe="")


@dataclass(frozen=True)
class BadgeTemplate:
    """One README badge template loaded from metadata."""

    alt: str
    image: str
    target: str


@dataclass(frozen=True)
class BadgeMetadata:
    """Badge metadata plus the repository fields used by badge templates."""

    repo: RepoMetadata
    badges: tuple[BadgeTemplate, ...]


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{CANONICAL_METADATA_PATH} is missing `{key}`.")
    return value.strip()


def _load_canonical_repo_metadata(payload: dict[str, object]) -> RepoMetadata | None:
    canonical_keys = ("host", "owner", "repo", "branch_name")
    present = [key for key in canonical_keys if str(payload.get(key, "")).strip()]
    if not present:
        return None
    if len(present) != len(canonical_keys):
        missing = ", ".join(key for key in canonical_keys if key not in present)
        raise RuntimeError(
            f"{CANONICAL_METADATA_PATH} has partial repository metadata: {missing}"
        )
    remote_name = str(payload.get("remote_name", "origin")).strip() or "origin"
    return RepoMetadata(
        host=_required_text(payload, "host"),
        owner=_required_text(payload, "owner"),
        repo=_required_text(payload, "repo"),
        remote_name=remote_name,
        branch_name=_required_text(payload, "branch_name"),
    )


def _load_badges(payload: dict[str, object]) -> tuple[BadgeTemplate, ...]:
    raw_badges = payload.get("badges")
    if not isinstance(raw_badges, list) or not raw_badges:
        raise RuntimeError(f"{CANONICAL_METADATA_PATH} must define non-empty `badges`.")

    badges: list[BadgeTemplate] = []
    for index, raw_badge in enumerate(raw_badges):
        if not isinstance(raw_badge, dict):
            raise RuntimeError(f"Badge entry {index} must be an object.")
        badge_payload = {str(key): value for key, value in raw_badge.items()}
        badges.append(
            BadgeTemplate(
                alt=_required_text(badge_payload, "alt"),
                image=_required_text(badge_payload, "image"),
                target=_required_text(badge_payload, "target"),
            )
        )
    return tuple(badges)


def load_metadata(repo_root: Path) -> BadgeMetadata:
    payload = json.loads(
        (repo_root / CANONICAL_METADATA_PATH).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"{CANONICAL_METADATA_PATH} must contain a JSON object.")

    normalized_payload = {str(key): value for key, value in payload.items()}
    repo_metadata = _load_canonical_repo_metadata(normalized_payload)
    if repo_metadata is None:
        repo_metadata = resolve_repo_metadata(repo_root)
    return BadgeMetadata(repo=repo_metadata, badges=_load_badges(normalized_payload))


def _git_executable() -> str:
    git_bin = shutil.which("git")
    if git_bin is None:
        raise RuntimeError(
            "git executable is required to resolve README badge metadata."
        )
    return git_bin


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        [_git_executable(), "-c", f"safe.directory={repo_root.resolve()}", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_remote_name(repo_root: Path) -> str:
    try:
        upstream = _run_git(
            repo_root,
            "rev-parse",
            "--abbrev-ref",
            "--symbolic-full-name",
            "@{upstream}",
        )
    except subprocess.CalledProcessError:
        upstream = ""
    if upstream and "/" in upstream:
        return upstream.split("/", 1)[0]

    branch_name = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    try:
        configured_remote = _run_git(
            repo_root, "config", "--get", f"branch.{branch_name}.remote"
        )
    except subprocess.CalledProcessError:
        configured_remote = ""
    if configured_remote:
        return configured_remote

    remotes = _run_git(repo_root, "remote").splitlines()
    if "origin" in remotes:
        return "origin"
    if len(remotes) == 1 and remotes[0]:
        return remotes[0]
    raise RuntimeError("Could not determine which Git remote drives README badges.")


def _resolve_branch_name(repo_root: Path, remote_name: str) -> str:
    head_ref = f"refs/remotes/{remote_name}/HEAD"
    try:
        symbolic_ref = _run_git(repo_root, "symbolic-ref", head_ref)
    except subprocess.CalledProcessError:
        symbolic_ref = ""
    if symbolic_ref.startswith(f"refs/remotes/{remote_name}/"):
        return symbolic_ref.rsplit("/", 1)[-1]

    remote_description = _run_git(repo_root, "remote", "show", remote_name)
    for line in remote_description.splitlines():
        stripped = line.strip()
        if stripped.startswith("HEAD branch: "):
            return stripped.removeprefix("HEAD branch: ").strip()
    return _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")


def _parse_remote_url(remote_url: str) -> tuple[str, str, str]:
    normalized_url = remote_url.strip()
    if "://" in normalized_url:
        parsed = urlparse(normalized_url)
        host = parsed.hostname or ""
        path = parsed.path
    else:
        if ":" not in normalized_url:
            raise RuntimeError(f"Unsupported Git remote URL: {remote_url}")
        remote_host, remote_path = normalized_url.split(":", 1)
        host = remote_host.rsplit("@", 1)[-1]
        path = f"/{remote_path}"

    path_parts = [part for part in path.strip("/").split("/") if part]
    if len(path_parts) < 2 or not host:
        raise RuntimeError(f"Unsupported Git remote URL: {remote_url}")

    owner = path_parts[-2]
    repo = path_parts[-1].removesuffix(".git")
    return host, owner, repo


def resolve_repo_metadata(repo_root: Path) -> RepoMetadata:
    remote_name = _resolve_remote_name(repo_root)
    host, owner, repo = _parse_remote_url(
        _run_git(repo_root, "remote", "get-url", remote_name)
    )
    branch_name = _resolve_branch_name(repo_root, remote_name)
    return RepoMetadata(
        host=host,
        owner=owner,
        repo=repo,
        remote_name=remote_name,
        branch_name=branch_name,
    )


def render_badge_block(metadata: BadgeMetadata) -> str:
    lines = [BADGE_BLOCK_BEGIN]
    for badge in metadata.badges:
        image = badge.image.format(
            repo_path=metadata.repo.github_path,
            branch=metadata.repo.branch_query_value,
        )
        target = badge.target.format(
            repo_path=metadata.repo.github_path,
            branch=metadata.repo.branch_query_value,
        )
        lines.append(f"[![{badge.alt}]({image})]({target})")
    lines.append(BADGE_BLOCK_END)
    return "\n".join(lines)


def extract_badge_block(readme_text: str) -> str:
    start = readme_text.find(BADGE_BLOCK_BEGIN)
    end = readme_text.find(BADGE_BLOCK_END)
    if start == -1 or end == -1:
        raise RuntimeError("README.md is missing the generated badge markers.")
    end += len(BADGE_BLOCK_END)
    return readme_text[start:end]


def update_readme(repo_root: Path, write: bool) -> int:
    readme_path = repo_root / README_PATH
    original_text = readme_path.read_text(encoding="utf-8")
    badge_block = render_badge_block(load_metadata(repo_root))
    updated_text = original_text.replace(
        extract_badge_block(original_text), badge_block
    )

    if updated_text == original_text:
        return 0
    if not write:
        sys.stderr.write(
            "README.md badge block is out of date. Run "
            "`python3 tools/update_readme_badges.py --write`.\n"
        )
        return 1
    readme_path.write_text(
        updated_text + ("" if updated_text.endswith("\n") else "\n"),
        encoding="utf-8",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the generated README badge block."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Fail if README.md does not match the generated badge block.",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="Rewrite README.md with the generated badge block.",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing README.md.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return update_readme(Path(args.repo_root).resolve(), write=args.write)


if __name__ == "__main__":
    raise SystemExit(main())
