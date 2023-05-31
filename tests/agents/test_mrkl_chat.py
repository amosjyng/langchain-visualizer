import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain import (
    LLMMathChain,
    OpenAI,
    SerpAPIWrapper,
    SQLDatabase,
    SQLDatabaseChain,
)
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/agents/implementations/mrkl_chat.html


@vcr.use_cassette()
async def mrkl_chat_demo():
    llm = ChatOpenAI(temperature=0)
    llm1 = OpenAI(temperature=0)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm1, verbose=True)
    db = SQLDatabase.from_uri("sqlite:///tests/resources/Chinook.db")
    db_chain = SQLDatabaseChain.from_llm(llm=llm1, db=db, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description=(
                "useful for when you need to answer questions about current events. "
                "You should ask targeted questions"
            ),
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
        Tool(
            name="FooBar DB",
            func=db_chain.run,
            description=(
                "useful for when you need to answer questions about FooBar. Input "
                "should be in the form of a question containing full context"
            ),
        ),
    ]
    mrkl = initialize_agent(
        tools, llm, agent="chat-zero-shot-react-description", verbose=True
    )
    return mrkl.run(
        "What is the full name of the artist who recently released an album called "
        "'The Storm Before the Calm' and are they in the FooBar database? If so, what "
        "albums of theirs are in the FooBar database?"
    )


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(mrkl_chat_demo())
    assert "Jagged Little Pill" in result.strip()


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(mrkl_chat_demo)
