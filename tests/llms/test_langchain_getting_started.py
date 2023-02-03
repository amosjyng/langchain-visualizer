import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/prompts/getting_started.html


llm = OpenAI(model_name="text-ada-001", n=2, best_of=2, temperature=0)

# ================================== Execute example ==================================


@vcr.use_cassette()
async def getting_started_demo():
    return llm.generate(["Tell me a joke", "Tell me a poem"] * 2)


def test_llm_usage_succeeds():
    """
    Check that it works like a regular prompt.
    Also, record playback for easy visualization.
    """
    result = asyncio.get_event_loop().run_until_complete(getting_started_demo())
    assert len(result.generations) == 4
    assert len(result.generations[0]) == 2


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(getting_started_demo)
