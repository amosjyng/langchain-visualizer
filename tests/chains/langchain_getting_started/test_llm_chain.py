import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

from tests import vcr

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/chains/getting_started.html


llm = OpenAI(temperature=0)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)


# ================================== Execute example ==================================


@vcr.use_cassette()
async def llm_chain_demo():
    return chain.run("colorful socks")


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(llm_chain_demo())
    assert result.strip() == "Socktastic!"


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(llm_chain_demo)
