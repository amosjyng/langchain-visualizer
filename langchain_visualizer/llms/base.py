from typing import List, Optional

from ice.trace import add_fields
from langchain.callbacks.manager import CallbackManagerForLLMRun, Callbacks
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI, OpenAIChat
from langchain.schema import BaseMessage, ChatResult, LLMResult

from langchain_visualizer.hijacking import VisualizationWrapper, ice_hijack

MODEL_COST_MAP = {
    "text-davinci-003": 1,
    "gpt-3.5-turbo": 0.1,
}


class LlmVisualizer(VisualizationWrapper):
    async def run(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        result: LLMResult = self.og_fn(
            self.og_obj, prompts=prompts, stop=stop, callbacks=callbacks
        )
        if isinstance(self.og_obj, OpenAI):
            llm_output = result.llm_output or {}
            total_tokens = llm_output.get("token_usage", {}).get("total_tokens", 0)
            davinci_equivalent = int(
                MODEL_COST_MAP.get(self.og_obj.model_name, 0) * total_tokens
            )
            if davinci_equivalent > 0:
                add_fields(davinci_equivalent_tokens=davinci_equivalent)
        return result


class ChatLlmVisualizer(VisualizationWrapper):
    async def run(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        """Run the LLM on the given prompt and input."""
        result: ChatResult = self.og_fn(
            self.og_obj, messages=messages, stop=stop, run_manager=run_manager
        )
        if isinstance(self.og_obj, OpenAIChat) or isinstance(self.og_obj, ChatOpenAI):
            llm_output = result.llm_output or {}
            total_tokens = llm_output.get("token_usage", {}).get("total_tokens", 0)
            davinci_equivalent = int(
                MODEL_COST_MAP.get(self.og_obj.model_name, 0) * total_tokens
            )
            if davinci_equivalent > 0:
                add_fields(davinci_equivalent_tokens=davinci_equivalent)
        return result


ice_hijack(BaseLLM, "generate", LlmVisualizer)
ice_hijack(BaseChatModel, "_generate", ChatLlmVisualizer)
ice_hijack(ChatOpenAI, "_generate", ChatLlmVisualizer)
