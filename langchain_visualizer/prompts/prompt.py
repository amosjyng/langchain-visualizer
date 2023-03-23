from fvalues import F, FValue
from langchain.formatting import formatter as og_formatter
from langchain.prompts.base import DEFAULT_FORMATTER_MAPPING


def new_format(format_string, /, *args, **kwargs):
    # if there are any issues with the formatting, let the formatter expose them first
    result = og_formatter.format(format_string, *args, **kwargs)
    parts = []
    # modified from string._vformat
    for literal_text, field_name, format_spec, _ in og_formatter.parse(format_string):
        if literal_text:
            parts.append(literal_text)

        if field_name is not None:
            obj, _ = og_formatter.get_field(field_name, args, kwargs)
            parts.append(
                FValue(
                    source=field_name,
                    value=obj,
                    formatted=og_formatter.format_field(obj, format_spec),
                )
            )

    return F(result, parts=tuple(parts))


DEFAULT_FORMATTER_MAPPING["f-string"] = new_format
