from __future__ import annotations

from pathlib import Path


def shell_quote(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    return "'" + text.replace("'", "'\"'\"'") + "'"


def double_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def rsync_ssh_options(port: int, key_path: str | None) -> str:
    ssh_parts = ["ssh", "-p", str(port)]
    if key_path:
        ssh_parts.extend(["-i", shell_quote(key_path)])
    return f"-e {double_quote(' '.join(ssh_parts))}"


def rsync_upload_command(
    source: str | Path,
    destination: str,
    *,
    port: int,
    key_path: str | None,
    delete: bool = False,
    excludes: list[str] | None = None,
) -> str:
    flags = "rsync -az"
    if delete:
        flags += " --delete"
    exclude_args = " ".join(f"--exclude {shell_quote(pattern)}" for pattern in (excludes or []))
    exclude_part = f" {exclude_args}" if exclude_args else ""
    return f"{flags} {rsync_ssh_options(port, key_path)}{exclude_part} {shell_quote(source)} {destination}"


def rsync_download_command(
    source: str,
    destination: str | Path,
    *,
    port: int,
    key_path: str | None,
) -> str:
    return f"rsync -az {rsync_ssh_options(port, key_path)} {source} {shell_quote(destination)}/"
