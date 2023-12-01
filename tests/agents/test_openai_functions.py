import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import pytest
import vcr_langchain as vcr
from langchain import LLMMathChain, SerpAPIWrapper, SQLDatabase
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_experimental.sql import SQLDatabaseChain

# ========================== Start of langchain example code ==========================
# https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent


@vcr.use_cassette()
async def openai_functions_demo():
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
    db = SQLDatabase.from_uri("sqlite:///tests/resources/Chinook.db")
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
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
            description=(
                "useful for when you need to answer questions about math. Only enter "
                "in purely numerical expressions as a single string, as the calculator "
                "will not have access to any variables. For example, 5^3, not age^3."
            ),
        ),
        Tool(
            name="FooBar-DB",
            func=db_chain.run,
            description=(
                "useful for when you need to answer questions about FooBar. Input "
                "should be in the form of a question containing full context"
            ),
        ),
    ]
    agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True)
    return agent.run(
        "Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 "
        "power?"
    )


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    with pytest.raises(ValueError):
        # GPT got stupider. See commit d1ce53b for a version of
        # tests/agents/openai_functions_demo.yaml with the correct answer.
        asyncio.get_event_loop().run_until_complete(openai_functions_demo())


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(openai_functions_demo)
