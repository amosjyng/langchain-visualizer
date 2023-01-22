from langchain.python import PythonREPL
from langchain.serpapi import SerpAPIWrapper

from langchain_visualizer.hijacking import ice_hijack

ice_hijack(SerpAPIWrapper, "run")
ice_hijack(PythonREPL, "run")
