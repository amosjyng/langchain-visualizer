from typing import Any, List, Union

from ice import json_value
from langchain.schema import LLMResult

og_json_value = json_value.to_json_value


def to_json_value(x: Any) -> json_value.JSONValue:
    if isinstance(x, LLMResult):
        generations = x.generations
        texts: Union[List[List[str]], List[str], str] = [
            g.text for sublist in generations for g in sublist
        ]
        if len(texts) == 1:
            texts = texts[0]
        if len(texts) == 1:
            # do it a second time because it's a list of lists
            texts = texts[0]
        return og_json_value(texts)

    return og_json_value(x)


json_value.to_json_value = to_json_value

# this import cannot happen earlier because otherwise ice.json_value.to_json_value
# does not get successfully overwritten
from ice.recipe import recipe  # noqa: E402

visualize = recipe.main
