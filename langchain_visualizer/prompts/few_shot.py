import string
from typing import List, Optional, Union

from fvalues import F, FValue
from langchain import FewShotPromptTemplate

from langchain_visualizer.hijacking import hijack
from langchain_visualizer.prompts.prompt import format_f


class FBuilder:
    def __init__(self, segments: Optional[List[Union[F, str]]] = None):
        self.f_values = segments or []

    def add_segment(self, segment: Union[F, str]):
        if not hasattr(segment, "parts"):
            segment = F(segment, parts=(segment,))
        self.f_values.append(segment)

    def pop(self):
        return self.f_values.pop()

    def build(self):
        finaL_str = "".join(self.f_values)
        final_parts = []
        previous_was_str = False
        for f_value in self.f_values:
            for part in f_value.flatten().parts:
                if isinstance(part, FValue):
                    final_parts.append(part)
                    previous_was_str = False
                else:  # it's a simple str
                    if previous_was_str:  # concatenate the two strings
                        final_parts[-1] = final_parts[-1] + part
                    else:
                        final_parts.append(part)

                    previous_was_str = True

        return F(finaL_str, parts=tuple(final_parts))


def get_new_format(og_format):
    def new_format(self, *args, **kwargs) -> str:
        if self.template_format != "f-string":
            return og_format(*args, **kwargs)

        f_builder = FBuilder()
        if self.prefix:
            f_builder.add_segment(
                format_f(string.Formatter(), self.prefix, *args, **kwargs)
            )
            f_builder.add_segment(self.example_separator)

        examples = self._get_examples(**kwargs)
        # ignore nested templating for now
        for example in examples:
            example_f = self.example_prompt.format(**example)
            f_builder.add_segment(example_f)
            f_builder.add_segment(self.example_separator)
        f_builder.pop()  # pop last separator in case there's no suffix

        if self.suffix:
            f_builder.add_segment(self.example_separator)
            f_builder.add_segment(
                format_f(string.Formatter(), self.suffix, *args, **kwargs)
            )

        return f_builder.build()

    return new_format


hijack(FewShotPromptTemplate, "format", get_new_format)
