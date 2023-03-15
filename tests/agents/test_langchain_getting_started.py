import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.agents import initialize_agent, load_tools
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/agents/getting_started.html


@vcr.use_cassette()
async def search_agent_demo():
    llm = OpenAI(temperature=0)
    tools = load_tools(["serpapi", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=True
    )
    return agent.run(
        "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 "
        "power?"
    )


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(search_agent_demo())
    assert result.strip().endswith("raised to the 0.23 power is 2.169459462491557.")


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(search_agent_demo)
