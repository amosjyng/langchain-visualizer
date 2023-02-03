import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.chains import LLMCheckerChain
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/chains/examples/llm_checker.html


@vcr.use_cassette()
async def checker_chain_demo():
    llm = OpenAI(temperature=0.7)
    text = "What type of mammal lays the biggest eggs?"
    checker_chain = LLMCheckerChain(llm=llm, verbose=True)
    checker_chain.run(text)


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(checker_chain_demo())
    assert "The Southern Elephant Seal" in result


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(checker_chain_demo)
