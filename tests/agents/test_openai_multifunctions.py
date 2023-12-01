import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain import SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI

# ========================== Start of langchain example code ==========================
# https://python.langchain.com/docs/modules/agents/agent_types/openai_multi_functions_agent


@vcr.use_cassette()
async def openai_multifunctions_demo():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    search = SerpAPIWrapper()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description=(
                "Useful when you need to answer questions about current events. You "
                "should ask targeted questions."
            ),
        ),
    ]
    mrkl = initialize_agent(
        tools, llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
    )
    return mrkl.run("What is the weather in LA and SF?")


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = (
        asyncio.get_event_loop()
        .run_until_complete(openai_multifunctions_demo())
        .lower()
    )
    assert "the weather in los angeles" in result
    assert "in san francisco" in result


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(openai_multifunctions_demo)
