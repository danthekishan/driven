import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from driven.core.harness import HarnessState, StateManager
from driven.core.schemas import BranchInfo, Message


def _to_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]

    return str(value)


class InMemoryStateManager(StateManager):
    def __init__(self):
        self._store: dict[str, HarnessState] = {}

    async def load(self, run_id: str) -> HarnessState:
        if run_id in self._store:
            return self._store[run_id]

        state = HarnessState(state_id=run_id)
        self._store[run_id] = state
        return state

    async def save(self, state: HarnessState) -> HarnessState:
        self._store[state.state_id] = state
        return state


class LocalStateManager(StateManager):
    """File-based state manager with sensible default storage location.

    Default location: ~/.driven/runs
    """

    def __init__(self, root_dir: Path | None = None):
        self.root_dir = root_dir or (Path.home() / ".driven" / "runs")
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _state_path(self, state: HarnessState) -> Path:
        if state.branch:
            return self.root_dir / state.branch.parent_run_id / "branches" / f"{state.state_id}.json"
        return self.root_dir / f"{state.state_id}.json"

    def _serialize(self, state: HarnessState) -> dict[str, Any]:
        return {
            "state_id": state.state_id,
            "system": state.system,
            "prompt": state.prompt,
            "step": state.step,
            "messages": [asdict(msg) for msg in state.messages],
            "event_log": _to_json_safe(state.event_log),
            "done": state.done,
            "internal": _to_json_safe(state.internal),
            "branch": asdict(state.branch) if state.branch else None,
            "branches": [asdict(b) for b in state.branches],
        }

    def _deserialize(self, payload: dict[str, Any]) -> HarnessState:
        messages = [Message(**msg) for msg in payload.get("messages", [])]
        branch_data = payload.get("branch")
        branch = BranchInfo(**branch_data) if branch_data else None
        branches = [BranchInfo(**b) for b in payload.get("branches", [])]

        return HarnessState(
            state_id=payload.get("state_id", ""),
            system=payload.get("system", ""),
            prompt=payload.get("prompt", ""),
            step=int(payload.get("step", 0)),
            messages=messages,
            event_log=payload.get("event_log", []),
            done=bool(payload.get("done", False)),
            internal=payload.get("internal", {}),
            branch=branch,
            branches=branches,
        )

    async def load(self, run_id: str) -> HarnessState:
        async with self._lock:
            for parent_dir in self.root_dir.iterdir():
                branch_path = parent_dir / "branches" / f"{run_id}.json"
                if branch_path.exists():
                    raw = json.loads(branch_path.read_text())
                    state = self._deserialize(raw)
                    if not state.state_id:
                        state.state_id = run_id
                    return state

            path = self.root_dir / f"{run_id}.json"
            if not path.exists():
                return HarnessState(state_id=run_id)

            raw = json.loads(path.read_text())
            state = self._deserialize(raw)

            if not state.state_id:
                state.state_id = run_id

            return state

    async def save(self, state: HarnessState) -> HarnessState:
        path = self._state_path(state)
        tmp = path.with_suffix(".tmp")
        payload = self._serialize(state)

        async with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
            tmp.replace(path)

        return state

    async def delete(self, run_id: str) -> None:
        async with self._lock:
            path = self.root_dir / f"{run_id}.json"
            if path.exists():
                path.unlink()
                return

            for parent_dir in self.root_dir.iterdir():
                branch_path = parent_dir / "branches" / f"{run_id}.json"
                if branch_path.exists():
                    branch_path.unlink()
                    return
