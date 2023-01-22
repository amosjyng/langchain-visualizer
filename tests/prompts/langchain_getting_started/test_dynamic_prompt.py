import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

from fvalues import FValue
from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI
from langchain.prompts.example_selector import LengthBasedExampleSelector

from tests import vcr

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/prompts/getting_started.html


# Next, we specify the template to format the examples we have provided.
# We use the `PromptTemplate` class for this.
example_formatter_template = """
Word: {word}
Antonym: {antonym}\n
""".strip()
example_prompt = PromptTemplate(
    input_variables=["word", "antonym"],
    template=example_formatter_template,
)

# These are a lot of examples of a pretend task of creating antonyms.
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "energetic", "antonym": "lethargic"},
    {"word": "sunny", "antonym": "gloomy"},
    {"word": "windy", "antonym": "calm"},
]

# We'll use the `LengthBasedExampleSelector` to select the examples.
example_selector = LengthBasedExampleSelector(
    examples=examples, example_prompt=example_prompt, max_length=25
)

# We can now use the `example_selector` to create a `FewShotPromptTemplate`.
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the antonym of every input",
    suffix="Word: {input}\nAntonym: ",
    input_variables=["input"],
    example_separator="\n\n",
)

# We can now generate a prompt using the `format` method.
long_string = (
    "big and huge and massive and large and gigantic and tall and much much "
    "much much much bigger than everything else"
)
prompt = dynamic_prompt.format(input=long_string)


# ================================== Execute example ==================================


def test_prompt():
    assert prompt.parts == (
        "Give the antonym of every input\n\nWord: ",
        FValue(source="word", value="happy", formatted="happy"),
        "\nAntonym: ",
        FValue(source="antonym", value="sad", formatted="sad"),
        "\n\nWord: ",
        FValue(
            source="input",
            value="big and huge and massive and large and gigantic and tall "
            "and much much much much much bigger than everything else",
            formatted="big and huge and massive and large and gigantic and "
            "tall and much much much much much bigger than everything else",
        ),
        "\nAntonym: ",
    )


@vcr.use_cassette()
async def dynamic_prompt_demo():
    agent = OpenAI(model_name="text-ada-001", temperature=0)
    return agent(prompt)


def test_llm_usage_succeeds():
    """
    Check that it works like a regular prompt.
    Also, record playback for easy visualization.
    """
    result = asyncio.get_event_loop().run_until_complete(dynamic_prompt_demo())
    assert result.strip().startswith("small")


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(dynamic_prompt_demo)
