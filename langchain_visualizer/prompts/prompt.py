from fvalues import F, FValue
from langchain import PromptTemplate
from langchain.formatting import formatter

from langchain_visualizer.hijacking import hijack


def format_f(formatter, string, *args, **kwargs) -> F:
    # if there are any issues with the formatting, let the formatter expose them first
    result = formatter.format(string, *args, **kwargs)
    parts = []
    # modified from string._vformat
    for literal_text, field_name, format_spec, _ in formatter.parse(string):
        if literal_text:
            parts.append(literal_text)

        if field_name is not None:
            obj, _ = formatter.get_field(field_name, args, kwargs)
            parts.append(
                FValue(
                    source=field_name,
                    value=obj,
                    formatted=formatter.format_field(obj, format_spec),
                )
            )

    return F(result, parts=tuple(parts))


def get_new_format(og_format):
    def new_format(self, *args, **kwargs) -> str:
        if self.template_format != "f-string":
            return og_format(*args, **kwargs)

        return format_f(formatter, self.template, *args, **kwargs)

    return new_format


hijack(PromptTemplate, "format", get_new_format)
