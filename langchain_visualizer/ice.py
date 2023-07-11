import time
from typing import Any, List, Union
from warnings import warn

from ice import json_value, server
from ice import settings as ice_settings
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatResult,
    FunctionMessage,
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
        chat_messages: List[BaseMessage] = [
            chat_generation.message for chat_generation in chat_generations
        ]
        if len(chat_messages) == 1:
            return og_json_value(chat_messages[0])
        return og_json_value(chat_messages)
    elif isinstance(x, SystemMessage):
        return {
            "System": x.content,
        }
    elif isinstance(x, AIMessage):
        if "function_call" in x.additional_kwargs:
            function_call = x.additional_kwargs["function_call"]
            return {
                "AI (function call)": function_call,
            }
        else:
            return {
                "AI": x.content,
            }
    elif isinstance(x, HumanMessage):
        return {
            "Human": x.content,
        }
    elif isinstance(x, FunctionMessage):
        return {
            "Function": x.name,
            "Result": x.content,
        }
    elif isinstance(x, BaseMessage):
        warn(f"Unknown message type: {x.type}")

    return og_json_value(x)


def wait_until_server_running():
    start_time = time.time()
    while not server.is_server_running():
        if time.time() - start_time > server.ICE_WAIT_TIME:
            raise TimeoutError(
                f"Server didn't start within {server.ICE_WAIT_TIME} seconds"
            )
        time.sleep(0.1)


json_value.to_json_value = to_json_value
server.ICE_WAIT_TIME = 10  # type: ignore
server.wait_until_server_running = wait_until_server_running
ice_settings.settings.OUGHT_ICE_HOST = "127.0.0.1"
