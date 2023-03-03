from typing import List, Optional

from ice.trace import add_fields
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI, OpenAIChat
from langchain.schema import LLMResult

from langchain_visualizer.hijacking import VisualizationWrapper, ice_hijack

MODEL_COST_MAP = {
    "text-davinci-003": 1,
    "gpt-3.5-turbo": 0.1,
}


class LlmVisualizer(VisualizationWrapper):
    async def run(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        result = self.og_fn(self.og_obj, prompts=prompts, stop=stop)
        if isinstance(self.og_obj, OpenAI) or isinstance(self.og_obj, OpenAIChat):
            total_tokens = result.llm_output.get("token_usage", {}).get(
                "total_tokens", 0
            )
            davinci_equivalent = int(
                MODEL_COST_MAP.get(self.og_obj.model_name, 0) * total_tokens
            )
            if davinci_equivalent > 0:
                add_fields(davinci_equivalent_tokens=davinci_equivalent)
        return result


ice_hijack(BaseLLM, "generate", LlmVisualizer)
