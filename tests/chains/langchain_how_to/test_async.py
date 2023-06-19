import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# ========================== Start of langchain example code ==========================
# https://python.langchain.com/docs/modules/chains/how_to/async_chain


async def async_generate(chain: LLMChain):
    resp = await chain.arun(product="toothpaste")
    return resp.strip()


async def generate_concurrently():
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tasks = [async_generate(chain) for _ in range(5)]
    return await asyncio.gather(*tasks)


@vcr.use_cassette()
async def test_async_api_demo():
    return await generate_concurrently()


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(test_async_api_demo())
    assert len(result) == 5


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(test_async_api_demo)
