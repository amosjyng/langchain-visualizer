# due to the complexity of managing patched functions, this file tests the usage of
# langchain_visualizer without vcr-langchain

import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import pytest
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/agents/getting_started.html

llm = OpenAI(temperature=0)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)


# ================================== Execute example ==================================


async def search_agent_demo():
    # reload tools because otherwise they will not use the cassette
    agent.tools = load_tools(["serpapi", "llm-math"], llm=llm)
    return agent.run(
        "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 "
        "power?"
    )


@pytest.mark.network
def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(search_agent_demo())
    assert "raised to the 0.23 power is 2." in result.strip()


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(search_agent_demo)
