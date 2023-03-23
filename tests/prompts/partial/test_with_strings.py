import langchain_visualizer  # isort:skip  # noqa: F401

import vcr_langchain as vcr
from langchain import PromptTemplate
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/prompts/examples/partial.html


@vcr.use_cassette()
async def test_partial_with_strings():
    agent = OpenAI(model_name="text-ada-001", temperature=0)
    prompt = PromptTemplate.from_template("Why did the {animal} cross the {surface}?")
    partial_prompt = prompt.partial(surface="road")
    final_prompt = partial_prompt.format(animal="chicken")
    assert final_prompt == "Why did the chicken cross the road?"
    return agent(final_prompt)


# ================================== Execute example ==================================

if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(test_partial_with_strings)
