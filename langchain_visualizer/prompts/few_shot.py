from fvalues import F
from langchain import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

from langchain_visualizer.hijacking import hijack


def get_new_format(og_format):
    def new_format(self, *args, **kwargs) -> str:
        if self.template_format != "f-string":
            return og_format(*args, **kwargs)

        # copied from FewShotPromptTemplate.format
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        examples = [
            {k: e[k] for k in self.example_prompt.input_variables} for e in examples
        ]
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
