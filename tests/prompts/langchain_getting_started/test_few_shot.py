import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

from fvalues import FValue
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI

from tests import vcr

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/prompts/getting_started.html


# First, create the list of few shot examples.
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
]

# Next, we specify the template to format the examples we have provided.
# We use the `PromptTemplate` class for this.
example_formatter_template = """
Word: {word}
Antonym: {antonym}
""".strip()
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"], template=example_formatter_template
)

few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym: ",
    input_variables=["input"],
    example_separator="\n\n",
)

# We can now generate a prompt using the `format` method.
prompt = few_shot_prompt.format(input="big")


# ================================== Execute example ==================================


def test_prompt():
    assert prompt.parts == (
        "Give the antonym of every input\n\nWord: ",
        FValue(source="word", value="happy", formatted="happy"),
        "\nAntonym: ",
        FValue(source="antonym", value="sad", formatted="sad"),
        "\n\nWord: ",
        FValue(source="word", value="tall", formatted="tall"),
        "\nAntonym: ",
        FValue(source="antonym", value="short", formatted="short"),
        "\n\nWord: ",
        FValue(source="input", value="big", formatted="big"),
        "\nAntonym: ",
    )


@vcr.use_cassette()
async def few_shot_prompt_demo():
    agent = OpenAI(model_name="text-ada-001", temperature=0)
    return agent(prompt)


def test_llm_usage_succeeds():
    """
    Check that it works like a regular prompt.
    Also, record playback for easy visualization.
    """
    result = asyncio.get_event_loop().run_until_complete(few_shot_prompt_demo())
    assert result.strip().startswith("small")


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(few_shot_prompt_demo)
