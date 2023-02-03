import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/chains/generic/sequential_chains.html

# This is an LLMChain to write a synopsis given a title of a play and the era it is
# set in.
llm = OpenAI(temperature=0)
template = """
You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:
""".strip()  # noqa
prompt_template = PromptTemplate(input_variables=["title", "era"], template=template)
synopsis_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="synopsis")

# This is an LLMChain to write a review of a play given a synopsis.
llm = OpenAI(temperature=0)
template = """
You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

Play Synopsis:
{synopsis}
Review from a New York Times play critic of the above play:
""".strip()  # noqa
prompt_template = PromptTemplate(input_variables=["synopsis"], template=template)
review_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="review")

# This is the overall chain where we run these two chains in sequence.
overall_chain = SequentialChain(
    chains=[synopsis_chain, review_chain],
    input_variables=["era", "title"],
    # Here we return multiple variables
    output_variables=["synopsis", "review"],
    verbose=True,
)


# ================================== Execute example ==================================


@vcr.use_cassette()
async def sequential_chain_demo():
    return overall_chain(
        {"title": "Tragedy at sunset on the beach", "era": "Victorian England"}
    )


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(sequential_chain_demo())
    assert (
        result["synopsis"]
        .strip()
        .startswith(
            "Tragedy at Sunset on the Beach is a play set in Victorian England."
        )
    )
    assert (
        result["review"]
        .strip()
        .startswith("Tragedy at Sunset on the Beach is a captivating and heartbreaking")
    )


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(sequential_chain_demo)
