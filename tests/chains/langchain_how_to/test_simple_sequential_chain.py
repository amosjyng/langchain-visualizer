import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

from langchain import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI

from tests import vcr

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/chains/generic/sequential_chains.html

# This is an LLMChain to write a synopsis given a title of a play.
llm = OpenAI(temperature=0)
template = """
You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

Title: {title}
Playwright: This is a synopsis for the above play:
""".strip()  # noqa
prompt_template = PromptTemplate(input_variables=["title"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is an LLMChain to write a review of a play given a synopsis.
llm = OpenAI(temperature=0)
template = """
You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:
""".strip()  # noqa
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template)

# This is the overall chain where we run these two chains in sequence.
overall_chain = SimpleSequentialChain(
    chains=[synopsis_chain, review_chain], verbose=True
)


# ================================== Execute example ==================================


@vcr.use_cassette()
async def simple_sequential_chain_demo():
    return overall_chain.run("Tragedy at sunset on the beach")


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(simple_sequential_chain_demo())
    assert result.strip().startswith(
        "Tragedy at Sunset on the Beach is a powerful and moving story of love"
    )


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(simple_sequential_chain_demo)
