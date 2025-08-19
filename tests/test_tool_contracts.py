from openagent.core.base import BaseTool


def test_tool_schema_and_capabilities():
    class T(BaseTool):
        async def execute(self, input_data):
            raise NotImplementedError

    t = T(name="t", description="test", capabilities=["terminal"], schema={"type": "object"})
    s = t.get_schema()
    assert "capabilities" in s and "terminal" in s["capabilities"]
    assert "parameters" in s and s["parameters"]["type"] == "object"
