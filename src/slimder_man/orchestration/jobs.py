from __future__ import annotations

import subprocess
from dataclasses import dataclass
from collections.abc import Iterator
from typing import Protocol

from slimder_man.utils.hashing import redact_secret


@dataclass
class CommandResult:
    command: str
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def redacted(self) -> "CommandResult":
        return CommandResult(
            command=redact_secret(self.command),
            returncode=self.returncode,
            stdout=redact_secret(self.stdout),
            stderr=redact_secret(self.stderr),
        )


class CommandExecutor(Protocol):
    def run(self, command: str) -> CommandResult:
        ...


class SubprocessExecutor:
    def run(self, command: str) -> CommandResult:
        completed = subprocess.run(command, shell=True, capture_output=True, text=True)
        return CommandResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def stream(self, command: str) -> Iterator[str]:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            yield redact_secret(line.rstrip("\n"))
        returncode = process.wait()
        if returncode != 0:
            yield f"COMMAND_EXITED_WITH_{returncode}"
