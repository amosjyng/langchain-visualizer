import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/chains/getting_started.html

llm = OpenAI(temperature=0)
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)

chain = LLMChain(llm=llm, prompt=prompt)

second_prompt = PromptTemplate(
    input_variables=["company_name"],
    template="Write a catchphrase for the following company: {company_name}",
)
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)


# ================================== Execute example ==================================


@vcr.use_cassette()
async def sequential_chain_demo():
    return overall_chain.run("colorful socks")


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(sequential_chain_demo())
    assert result.strip() == '"Step into Comfort with Socktastic!"'


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(sequential_chain_demo)
