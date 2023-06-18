from langchain.chains.base import Chain

from langchain_visualizer.hijacking import ice_hijack

ice_hijack(Chain, "__call__")
ice_hijack(Chain, "acall")
