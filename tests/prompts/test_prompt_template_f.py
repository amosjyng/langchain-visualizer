# ensure compatibility with fvalues by modifying tests from
# https://github.com/oughtinc/fvalues/blob/4baf69e/tests/test_f.py
# langchain does not appear to support numerical format options, so we'll skip those
# tests

import langchain_visualizer  # isort:skip  # noqa: F401

from fvalues import FValue
from langchain import PromptTemplate


def test_add_f():
    f1 = PromptTemplate(template="hello {foo}", input_variables=["foo"]).format(foo="3")
    f2 = PromptTemplate(template="world {bar}", input_variables=["bar"]).format(bar="7")
    f3 = f1 + " " + f2
    assert f3 == "hello 3 world 7"
    assert f3.parts == (
        FValue(source='f1 + " "', value="hello 3 ", formatted="hello 3 "),
        FValue(source="f2", value="world 7", formatted="world 7"),
    )
    assert f3.flatten().parts == (
        "hello ",
        FValue(source="foo", value="3", formatted="3"),
        " ",
        "world ",
        FValue(source="bar", value="7", formatted="7"),
    )


def test_still_node_from_eval():
    # unlike the original fvalues, PromptTemplate should work regardless
    s = eval(
        'PromptTemplate(template="hello {foo}", '
        'input_variables=["foo"]).format(foo="world")'
    )
    assert s == "hello world"
    assert s.parts == (
        "hello ",
        FValue(source="foo", value="world", formatted="world"),
    )


def test_strip():
    space = " "
    s = PromptTemplate(
        template=" {space} hello {space} ", input_variables=["space"]
    ).format(space=space)
    assert s == "   hello   "
    assert s.parts == (
        " ",
        FValue(source="space", value=" ", formatted=" "),
        " hello ",
        FValue(source="space", value=" ", formatted=" "),
        " ",
    )
    assert s.strip() == "hello"
    assert s.strip(space) == "hello"
    assert s.lstrip() == "hello   "
    assert s.lstrip(space) == "hello   "
    assert s.rstrip() == "   hello"
    assert s.rstrip(space) == "   hello"
    assert s.strip().parts == ("hello",)
    assert s.lstrip().parts == (
        "hello ",
        FValue(source="space", value=" ", formatted=" "),
        " ",
    )
    assert s.rstrip().parts == (
        " ",
        FValue(source="space", value=" ", formatted=" "),
        " hello",
    )
    assert s.strip().strip("ho").strip() == "ell"
