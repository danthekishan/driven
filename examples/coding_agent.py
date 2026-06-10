from pydantic import BaseModel

from driven.core.protocols import HarnessState
from driven.core.tool import tool
from driven.core.tool_runtime import SubAgent
from coder import CoderExtension


class CodeTaskInput(BaseModel):
    task: str


class CodingAgent(SubAgent):
    name = "coding-agent"
    description = "Spawns a coding agent as a branch to perform tasks."
    private_extensions = [CoderExtension(workspace="./workspace")]

    @tool(description="Run a coding task", public=True)
    async def run(
        self,
        input: CodeTaskInput,
        state: HarnessState,
    ) -> dict:
        branch_state, error = await self.branch(
            prompt=input.task,
            system=(
                "You are a coding agent. "
                "Perform the requested coding task using the available tools. "
                "When done, summarize what you did."
            ),
            parent_state=state,
        )

        if error:
            return {
                "task": input.task,
                "status": "error",
                "error": str(error),
            }

        if branch_state and branch_state.messages:
            last_msg = branch_state.messages[-1]
            return {
                "task": input.task,
                "status": "completed",
                "summary": last_msg.content,
                "steps": branch_state.step,
                "branch_id": (
                    branch_state.branch.branch_id if branch_state.branch else None
                ),
            }

        return {
            "task": input.task,
            "status": "no_output",
        }
