import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.chains import LLMBashChain
from langchain.llms import OpenAI

llm = OpenAI(temperature=0)
chain = LLMBashChain.from_llm(llm=llm)


# ================================== Execute example ==================================


@vcr.use_cassette()
async def bash_chain_demo():
    return chain("What files are in my current directory?")


def test_bash_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(bash_chain_demo())
    assert "langchain_visualizer" in result["answer"]


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(bash_chain_demo)
