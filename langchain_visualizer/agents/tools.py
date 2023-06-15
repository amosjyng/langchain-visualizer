from langchain.python import PythonREPL
from langchain.serpapi import SerpAPIWrapper
from langchain.sql_database import SQLDatabase
from langchain.tools.playwright.click import ClickTool
from langchain.tools.playwright.current_page import CurrentWebPageTool
from langchain.tools.playwright.extract_hyperlinks import ExtractHyperlinksTool
from langchain.tools.playwright.extract_text import ExtractTextTool
from langchain.tools.playwright.get_elements import GetElementsTool
from langchain.tools.playwright.navigate import NavigateTool
from langchain.tools.playwright.navigate_back import NavigateBackTool
from langchain.utilities.bash import BashProcess

from langchain_visualizer.hijacking import ice_hijack

ice_hijack(SerpAPIWrapper, "run")
ice_hijack(PythonREPL, "run")
ice_hijack(BashProcess, "run")
ice_hijack(SQLDatabase, "run")

ice_hijack(ClickTool, "arun")
ice_hijack(CurrentWebPageTool, "arun")
ice_hijack(ExtractHyperlinksTool, "arun")
ice_hijack(ExtractTextTool, "arun")
ice_hijack(GetElementsTool, "arun")
ice_hijack(NavigateTool, "arun")
ice_hijack(NavigateBackTool, "arun")
