import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain.llms import OpenAI

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/memory/getting_started.html


llm = OpenAI(temperature=0)
conversation_with_summary = ConversationChain(
    llm=llm, memory=ConversationSummaryMemory(llm=OpenAI()), verbose=True
)


# ================================== Execute example ==================================


@vcr.use_cassette()
async def conversation_summary_memory_demo():
    conversation_with_summary.predict(input="Hi, what's up?")
    conversation_with_summary.predict(input="Tell me more about it!")
    return conversation_with_summary.predict(
        input="Very cool -- what is the scope of the project?"
    )


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(
        conversation_summary_memory_demo()
    )
    assert result.strip().startswith("The scope of the project is to")


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(conversation_summary_memory_demo)
