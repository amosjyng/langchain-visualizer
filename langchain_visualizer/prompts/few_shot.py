from typing import Iterable

from fvalues import F
from langchain import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

from langchain_visualizer.hijacking import hijack


def preserved_join(self, iterable: Iterable[str]) -> "F":
    """Copied from https://github.com/oughtinc/fvalues/pull/11 pending merge."""
    joined = ""
    parts = []
    for substring in iterable:
        joined += substring
        parts.append(substring)

        # avoid polluting parts when joining with empty string
        if str(self) != "":
            joined += self
            parts.append(self)

    if len(parts) > 0 and str(self) != "":  # pop the last joiner
        joined = joined[: -len(self)]
        parts.pop()

    return F(joined, parts=tuple(parts))


setattr(F, "preserved_join", preserved_join)


def get_new_format(og_format):
    def new_format(self, *args, **kwargs) -> str:
        if self.template_format != "f-string":
            return og_format(*args, **kwargs)

        # copied from FewShotPromptTemplate.format
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        # Format the examples.
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]
        # Create the overall template.
        prefix_template = PromptTemplate.from_template(self.prefix)
        suffix_template = PromptTemplate.from_template(self.suffix)
        prefix_args = {
            k: v for k, v in kwargs.items() if k in prefix_template.input_variables
        }
        suffix_args = {
            k: v for k, v in kwargs.items() if k in suffix_template.input_variables
        }
        pieces = [
            prefix_template.format(**prefix_args),
            *example_strings,
            suffix_template.format(**suffix_args),
        ]
        return F(self.example_separator).preserved_join(  # type: ignore
            [piece for piece in pieces if piece]
        )

    return new_format


hijack(FewShotPromptTemplate, "format", get_new_format)
