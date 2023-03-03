import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/prompts/getting_started.html


llm = OpenAI(model_name="gpt-3.5-turbo")

# ================================== Execute example ==================================


@vcr.use_cassette()
async def chatgpt_demo():
    return llm("Write me 5 paragraphs on why technological progress is necessary")


def test_llm_usage_succeeds():
    """
    Check that it works like a regular prompt.
    Also, record playback for easy visualization.
    """
    result = asyncio.get_event_loop().run_until_complete(chatgpt_demo())
    assert result.endswith("for generations to come.")


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(chatgpt_demo)
