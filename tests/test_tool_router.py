from openagent.core.agent import Agent
from openagent.core.base import BaseTool


class Dummy(BaseTool):
    async def execute(self, input_data):
        raise NotImplementedError


def test_relevance_simple():
    agent = Agent(
        name="A",
        tools=[
            Dummy(name="git_tool", description="...", capabilities=["terminal"]),
            Dummy(name="file_manager", description="...", capabilities=["terminal"]),
        ],
        model_name="tiny-llama",
    )
    assert agent._tool_is_relevant_sync(agent.get_tool("git_tool"), "show git status")
    assert not agent._tool_is_relevant_sync(agent.get_tool("git_tool"), "what is the weather?")
    assert agent._tool_is_relevant_sync(agent.get_tool("file_manager"), "list files")
