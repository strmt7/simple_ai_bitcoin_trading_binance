#!/usr/bin/env python3
"""Push to GitHub over HTTPS using a one-shot, socket-backed PAT helper.

The token is read from the ``GITHUB_TOKEN`` environment variable (or prompted
for interactively with ``getpass``) and served to ``git push`` via a scoped
``GIT_ASKPASS`` program that fetches the token from a short-lived UNIX socket.

The token therefore never appears in:

* ``argv`` of any process (nothing is ever passed on the command line),
* the ``origin`` URL (``credential.helper`` is suppressed for this call),
* the user's ``~/.git-credentials`` (ditto),
* shell history (no heredoc, no export of the token by the user),
* disk files owned by git (the askpass script is mode 0700 under /tmp).

This is an adapted descendant of the ZMB-UZH/omero-docker-extended
``tools/git_push_with_pat.py`` helper, trimmed for the trading-CLI repo.
"""

from __future__ import annotations

import argparse
import getpass
import os
import shutil
import socket
import stat
import subprocess  # nosec B404
import sys
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from urllib.parse import urlparse


RunCommand = Callable[..., "subprocess.CompletedProcess[str]"]
TokenReader = Callable[[str], str]
_ALLOWED_GITHUB_OWNER = "strmt7"
_ALLOWED_GITHUB_REPO = "simple_ai_bitcoin_trading_binance"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run git push with a socket-backed askpass helper so PATs never "
            "appear in argv, remotes, logs, or long-lived credential stores."
        )
    )
    parser.add_argument("remote", help="Git remote name or URL.")
    parser.add_argument("refspec", help="Branch or refspec to push.")
    parser.add_argument(
        "--username",
        default="x-access-token",
        help="Username supplied to GitHub's HTTPS prompt.",
    )
    parser.add_argument(
        "--token-env",
        default="GITHUB_TOKEN",
        help="Env var from which to read the PAT before prompting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the authentication path without touching the remote.",
    )
    return parser.parse_args(argv)


def _validate_git_argument(name: str, value: str) -> None:
    if not value or value.startswith("-") or "\x00" in value:
        raise SystemExit(f"{name} must be a non-option git argument")
    if any(ord(character) < 32 for character in value):
        raise SystemExit(f"{name} must not contain control characters")


def _read_token(env: Mapping[str, str], env_name: str, reader: TokenReader) -> str:
    token = env.get(env_name, "").strip()
    if token:
        return token
    if not sys.stdin.isatty():
        raise SystemExit(f"{env_name} is required")
    token = reader("GitHub PAT: ").strip()
    if not token:
        raise SystemExit(f"{env_name} is required")
    return token


def _remote_url_from_argument(
    remote: str,
    *,
    git_bin: str,
    env: Mapping[str, str],
    runner: RunCommand,
) -> str:
    if "://" in remote or remote.startswith("git@"):
        return remote
    result = runner(
        [git_bin, "remote", "get-url", remote],
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
    )
    if int(getattr(result, "returncode", 1)) != 0:
        raise SystemExit(f"unable to resolve git remote {remote!r}")
    return str(getattr(result, "stdout", "")).strip()


def _github_owner_repo(remote_url: str) -> tuple[str, str] | None:
    if remote_url.startswith("git@github.com:"):
        return None
    parsed = urlparse(remote_url)
    if parsed.scheme != "https" or parsed.hostname not in {"github.com", "www.github.com"}:
        return None
    parts = parsed.path.removesuffix(".git").strip("/").split("/")
    if len(parts) != 2:
        return None
    return parts[0].lower(), parts[1].lower()


def _validate_allowed_remote(remote_url: str) -> None:
    owner_repo = _github_owner_repo(remote_url)
    if owner_repo != (_ALLOWED_GITHUB_OWNER, _ALLOWED_GITHUB_REPO):
        raise SystemExit(
            "refusing to serve a GitHub PAT to an unexpected remote; "
            f"expected github.com/{_ALLOWED_GITHUB_OWNER}/{_ALLOWED_GITHUB_REPO}"
        )


def _write_askpass(path: Path) -> None:
    executable = sys.executable or "/usr/bin/env python3"
    script = "\n".join([
        f"#!{executable}",
        "from __future__ import annotations",
        "",
        "import os",
        "import socket",
        "import sys",
        "",
        "prompt = sys.argv[1] if len(sys.argv) > 1 else ''",
        "if 'sername' in prompt:",
        "    username = os.environ.get('GIT_PAT_USERNAME', '')",
        "    if not username:",
        "        raise SystemExit(1)",
        "    print(username)",
        "elif 'assword' in prompt:",
        "    socket_path = os.environ.get('GIT_PAT_SOCKET', '')",
        "    if not socket_path:",
        "        raise SystemExit(1)",
        "    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:",
        "        client.connect(socket_path)",
        "        chunks = []",
        "        while True:",
        "            chunk = client.recv(4096)",
        "            if not chunk:",
        "                break",
        "            chunks.append(chunk)",
        "    if not chunks:",
        "        raise SystemExit(1)",
        "    sys.stdout.buffer.write(b''.join(chunks))",
        "else:",
        "    raise SystemExit(1)",
        "",
    ])
    path.write_text(script, encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def _serve_credential_once(socket_path: Path, credential: str):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    socket_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    server.listen(1)
    server.settimeout(0.25)

    stop = threading.Event()
    payload = f"{credential}\n".encode("utf-8")

    def serve() -> None:
        try:
            while not stop.is_set():
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.sendall(payload)
                return
        finally:
            server.close()

    thread = threading.Thread(target=serve, name="git-pat-askpass", daemon=True)
    thread.start()
    return stop, thread


def run_push(
    args: argparse.Namespace,
    *,
    env: Mapping[str, str] | None = None,
    token_reader: TokenReader = getpass.getpass,
    runner: RunCommand = subprocess.run,
) -> int:
    _validate_git_argument("remote", args.remote)
    _validate_git_argument("refspec", args.refspec)
    _validate_git_argument("username", args.username)

    git_bin = shutil.which("git")
    if git_bin is None:
        raise SystemExit("git is required")
    base_env = dict(os.environ if env is None else env)
    remote_url = _remote_url_from_argument(args.remote, git_bin=git_bin, env=base_env, runner=runner)
    _validate_allowed_remote(remote_url)
    token = _read_token(base_env, args.token_env, token_reader)

    temp_root = Path(tempfile.mkdtemp(prefix="git-pat-askpass-"))
    temp_root.chmod(stat.S_IRWXU)
    askpass_path = temp_root / "askpass.py"
    socket_path = temp_root / "credential.sock"
    stop_server = None
    server_thread = None
    try:
        _write_askpass(askpass_path)
        stop_server, server_thread = _serve_credential_once(socket_path, token)
        push_env = base_env.copy()
        push_env.pop(args.token_env, None)
        push_env.update({
            "GIT_ASKPASS": str(askpass_path),
            "GIT_PAT_SOCKET": str(socket_path),
            "GIT_PAT_USERNAME": args.username,
            "GIT_TERMINAL_PROMPT": "0",
        })
        command = [
            git_bin,
            "-c", "credential.helper=",
            "-c", "credential.https://github.com.helper=",
            "push",
        ]
        if args.dry_run:
            command.append("--dry-run")
        command.extend([args.remote, args.refspec])
        result = runner(command, env=push_env, check=False)
        return int(getattr(result, "returncode", 0))
    finally:
        if stop_server is not None:
            stop_server.set()
        if server_thread is not None:
            server_thread.join(timeout=2)
        shutil.rmtree(temp_root, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    return run_push(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
