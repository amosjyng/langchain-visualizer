from typing import Any, List, Union
import subprocess
import sys
import os
import selectors
import time

from ice import json_value, server
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


def wait_until_server_running():
    start_time = time.time()
    while not server.is_server_running():
        if time.time() - start_time > server.ICE_WAIT_TIME:
            raise TimeoutError(f"Server didn't start within {server.ICE_WAIT_TIME} seconds")
        time.sleep(0.1)


def ensure_server_running():
    if server.is_server_running():
        return

    server.log.info("Starting server, set OUGHT_ICE_AUTO_SERVER=0 to disable.")
    try:
        server_process = subprocess.Popen(
            [sys.executable, "-m", "ice.server", "start"],
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        sel = selectors.DefaultSelector()
        sel.register(server_process.stdout, selectors.EVENT_READ)
        server.wait_until_server_running()
        server.log.info("Server started! Run `python -m ice.server stop` to stop it.")
    except TimeoutError as e:
        for key, _ in sel.select(timeout=0):
            output = key.fileobj.readline().decode('utf-8').strip()
        server.log.error(f"ICE server failed to start. Command output: {output}")
        raise e


json_value.to_json_value = to_json_value
server.ICE_WAIT_TIME = 10
server.wait_until_server_running = wait_until_server_running
server.ensure_server_running = ensure_server_running
