from typing import Any, List, Optional, Union

from ice.trace import add_fields
from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
    Callbacks,
)
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI, OpenAIChat
from langchain.schema import BaseMessage, ChatResult, LLMResult

from langchain_visualizer.hijacking import VisualizationWrapper, ice_hijack

# todo: ideally we would stop tying costs to Davinci tokens, but this would involve
# changing ICE's frontend logic
MODEL_COST_MAP = {
    # model names: https://platform.openai.com/docs/models/gpt-3-5
    # model costs: https://openai.com/pricing
    "text-davinci-003": 1,
    "gpt-3.5-turbo": 0.1,
    "gpt-3.5-turbo-16k": 0.2,
}


class LlmVisualizer(VisualizationWrapper):
    def determine_cost(self, result: Union[LLMResult, ChatResult]) -> None:
        llm_output = result.llm_output or {}
        total_tokens = llm_output.get("token_usage", {}).get("total_tokens", 0)
        davinci_equivalent = int(
            MODEL_COST_MAP.get(self.og_obj.model_name, 0) * total_tokens
        )
        if davinci_equivalent > 0:
            add_fields(davinci_equivalent_tokens=davinci_equivalent)


class LlmSyncVisualizer(LlmVisualizer):
    """Overrides the sync generate function for regular LLMs."""

    async def run(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        result: LLMResult = self.og_fn(
            self.og_obj, prompts=prompts, stop=stop, callbacks=callbacks, **kwargs
        )
        if isinstance(self.og_obj, OpenAI):
            self.determine_cost(result)
        return result


class LlmAsyncVisualizer(LlmVisualizer):
    """Overrides the async generate function for regular LLMs."""

    async def run(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Callbacks = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        result: LLMResult = await self.og_fn(
            self.og_obj,
            prompts=prompts,
            stop=stop,
            callbacks=callbacks,
            **kwargs,
        )
        if isinstance(self.og_obj, OpenAI):
            self.determine_cost(result)
        return result


class ChatLlmSyncVisualizer(LlmVisualizer):
    """Overrides the sync _generate function for chat LLMs."""

    async def run(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Run the LLM on the given prompt and input."""
        result: ChatResult = self.og_fn(
            self.og_obj, messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )
        if isinstance(self.og_obj, OpenAIChat) or isinstance(self.og_obj, ChatOpenAI):
            self.determine_cost(result)
        return result


class ChatLlmAsyncVisualizer(LlmVisualizer):
    """Overrides the async _agenerate function for chat LLMs."""

    async def run(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        """Run the LLM on the given prompt and input."""
        result: ChatResult = await self.og_fn(
            self.og_obj,
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        if isinstance(self.og_obj, OpenAIChat) or isinstance(self.og_obj, ChatOpenAI):
            self.determine_cost(result)
        return result


ice_hijack(BaseLLM, "generate", LlmSyncVisualizer)
ice_hijack(BaseLLM, "agenerate", LlmAsyncVisualizer)
ice_hijack(ChatOpenAI, "_generate", ChatLlmSyncVisualizer)
ice_hijack(ChatOpenAI, "_agenerate", ChatLlmAsyncVisualizer)
