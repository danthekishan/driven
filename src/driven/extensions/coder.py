import asyncio
import difflib
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from driven.core.harness import Emitter
from driven.core.tool_runtime import Extension, tool


# =========================
# INPUT SCHEMAS
# =========================


class ReadFileInput(BaseModel):
    path: str


class WriteFileInput(BaseModel):
    path: str
    content: str


class AppendFileInput(BaseModel):
    path: str
    content: str


class ListDirectoryInput(BaseModel):
    path: str = "."


class SearchFilesInput(BaseModel):
    query: str
    path: str = "."


class RunCommandInput(BaseModel):
    command: str
    cwd: str = "."
    timeout: float = 30.0


class DiffFilesInput(BaseModel):
    old_path: str
    new_path: str


# =========================
# CODER EXTENSION
# =========================


class CoderExtension(Extension):
    name = "coder"
    description = "Coding and filesystem tools."

    def __init__(
        self,
        workspace: str = ".",
    ):
        self.workspace = Path(workspace).resolve()

        super().__init__()

    async def start(self):
        self.workspace.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =========================
    # HELPERS
    # =========================

    def _resolve_path(
        self,
        path: str,
    ) -> Path:
        resolved = (self.workspace / path).resolve()

        # sandbox protection
        if not str(resolved).startswith(str(self.workspace)):
            raise RuntimeError("path escapes workspace")

        return resolved

    # =========================
    # FILESYSTEM TOOLS
    # =========================

    @tool(description="Read a file from the workspace.")
    async def read_file(
        self,
        input: ReadFileInput,
    ) -> str:
        path = self._resolve_path(input.path)

        if not path.exists():
            raise RuntimeError(f"file does not exist: {input.path}")

        return path.read_text()

    @tool(description="Write content to a file.")
    async def write_file(
        self,
        input: WriteFileInput,
    ) -> dict:
        path = self._resolve_path(input.path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(input.content)

        return {
            "path": input.path,
            "bytes_written": len(input.content),
        }

    @tool(description="Append content to a file.")
    async def append_file(
        self,
        input: AppendFileInput,
    ) -> dict:
        path = self._resolve_path(input.path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open("a") as f:
            f.write(input.content)

        return {
            "path": input.path,
            "bytes_appended": len(input.content),
        }

    @tool(description="List files and directories.")
    async def list_directory(
        self,
        input: ListDirectoryInput,
    ) -> list[str]:
        path = self._resolve_path(input.path)

        if not path.exists():
            raise RuntimeError(f"path does not exist: {input.path}")

        items: list[str] = []

        for child in path.iterdir():
            relative = child.relative_to(self.workspace)

            if child.is_dir():
                items.append(f"{relative}/")
            else:
                items.append(str(relative))

        return sorted(items)

    @tool(description="Search files for text.")
    async def search_files(
        self,
        input: SearchFilesInput,
    ) -> list[dict]:
        base = self._resolve_path(input.path)

        results: list[dict] = []

        for file in base.rglob("*"):
            if not file.is_file():
                continue

            try:
                content = file.read_text()

            except Exception:
                continue

            lines = content.splitlines()

            for idx, line in enumerate(lines):
                if input.query.lower() in line.lower():
                    results.append(
                        {
                            "path": str(file.relative_to(self.workspace)),
                            "line": idx + 1,
                            "content": line.strip(),
                        }
                    )

        return results

    @tool(description="Show diff between two files.")
    async def diff_files(
        self,
        input: DiffFilesInput,
    ) -> str:
        old_path = self._resolve_path(input.old_path)

        new_path = self._resolve_path(input.new_path)

        old_lines = old_path.read_text().splitlines()

        new_lines = new_path.read_text().splitlines()

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=input.old_path,
            tofile=input.new_path,
            lineterm="",
        )

        return "\n".join(diff)

    # =========================
    # SHELL TOOL
    # =========================

    @tool(description=("Run a shell command inside the workspace."))
    async def run_command(
        self,
        input: RunCommandInput,
        emitter: Optional[Emitter] = None,
    ) -> dict:
        if emitter:
            await emitter.emit(
                "command.started",
                {
                    "command": input.command,
                    "cwd": input.cwd,
                },
            )

        cwd = self._resolve_path(input.cwd)

        try:
            proc = await asyncio.create_subprocess_shell(
                input.command,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=input.timeout,
            )

        except TimeoutError:
            raise RuntimeError(f"command timed out after {input.timeout}s")

        stdout_text = stdout.decode()
        stderr_text = stderr.decode()

        result = {
            "command": input.command,
            "returncode": proc.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }

        if emitter:
            await emitter.emit(
                "command.completed",
                result,
            )

        return result
