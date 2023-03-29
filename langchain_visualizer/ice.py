from typing import Any, List, Union

from ice import json_value
from langchain.schema import (
    AIMessage,
    ChatResult,
    HumanMessage,
    LLMResult,
    SystemMessage,
)

og_json_value = json_value.to_json_value


def to_json_value(x: Any) -> json_value.JSONValue:
    if isinstance(x, LLMResult):
        regular_generations = x.generations
        regular_texts: Union[List[List[str]], List[str], str] = [
            g.text for sublist in regular_generations for g in sublist
        ]
        if len(regular_texts) == 1:
            regular_texts = regular_texts[0]
        if len(regular_texts) == 1:
            # do it a second time because it's a list of lists
            regular_texts = regular_texts[0]
        return og_json_value(regular_texts)
    elif isinstance(x, ChatResult):
        chat_generations = x.generations
        chat_texts: Union[List[str], str] = [
            chat_generation.text for chat_generation in chat_generations
        ]
        if len(chat_texts) == 1:
            regular_texts = chat_texts[0]
        return og_json_value(chat_texts)
    elif isinstance(x, SystemMessage):
        return {
            "System": x.content,
        }
    elif isinstance(x, AIMessage):
        return {
            "AI": x.content,
        }
    elif isinstance(x, HumanMessage):
        return {
            "Human": x.content,
        }

    return og_json_value(x)


json_value.to_json_value = to_json_value
