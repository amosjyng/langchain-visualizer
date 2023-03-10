import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/chat/getting_started.html


@vcr.use_cassette()
async def chatgpt_demo():
    chat = ChatOpenAI(model_name="gpt-3.5-turbo")

    template = (
        "You are a helpful assistant that translates {input_language} to "
        "{output_language}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    return chat(
        chat_prompt.format_prompt(
            input_language="English",
            output_language="French",
            text="I love programming.",
        ).to_messages()
    )


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """
    Check that it works like a regular prompt.
    Also, record playback for easy visualization.
    """
    result = asyncio.get_event_loop().run_until_complete(chatgpt_demo())
    assert result.content == "J'adore programmer."


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(chatgpt_demo)
