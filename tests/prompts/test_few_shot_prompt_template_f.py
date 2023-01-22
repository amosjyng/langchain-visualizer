import langchain_visualizer  # isort:skip  # noqa: F401

from fvalues import FValue
from langchain import FewShotPromptTemplate, PromptTemplate


def test_few_shot_f():
    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
    ]

    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template="w={word},a={antonym}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="Give the antonym of every input:",
        suffix="w={input},a=",
        input_variables=["input"],
        example_separator="  ",
    )

    s = few_shot_prompt.format(input="big")
    assert (
        s == "Give the antonym of every input:  w=happy,a=sad  w=tall,a=short  w=big,a="
    )
    assert s.parts == (
        "Give the antonym of every input:  w=",
        FValue(source="word", value="happy", formatted="happy"),
        ",a=",
        FValue(source="antonym", value="sad", formatted="sad"),
        "  w=",
        FValue(source="word", value="tall", formatted="tall"),
        ",a=",
        FValue(source="antonym", value="short", formatted="short"),
        "  w=",
        FValue(source="input", value="big", formatted="big"),
        ",a=",
    )
