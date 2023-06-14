import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.agents import AgentType, initialize_agent
from langchain.agents.agent_toolkits import PlayWrightBrowserToolkit
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from vcr_langchain.dummy import get_async_test_browser

# ========================== Start of langchain example code ==========================
# https://python.langchain.com/en/latest/modules/agents/agents/examples/structured_chat.html


@vcr.use_cassette()
async def structured_tool_chat_demo() -> str:
    browser = get_async_test_browser(
        cassette_path="tests/agents/structured_tool_chat_demo.yaml"
    )
    browser_toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    tools = browser_toolkit.get_tools()

    llm = ChatOpenAI(temperature=0)
    chat_history = MessagesPlaceholder(variable_name="chat_history")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_kwargs={
            "memory_prompts": [chat_history],
            "input_variables": ["input", "agent_scratchpad", "chat_history"],
        },
    )
    agent_chain.run(input="Hi I'm Erica.")
    agent_chain.run(input="whats my name?")
    return await agent_chain.arun(
        input=(
            "What's the latest xkcd comic about? "
            "Navigate to the xkcd website and tell me."
        )
    )


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(structured_tool_chat_demo())
    assert result.strip().startswith("The latest xkcd comic is titled ")


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(structured_tool_chat_demo)
