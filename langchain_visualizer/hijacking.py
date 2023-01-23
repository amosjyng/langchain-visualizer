import asyncio

from ice.trace import TracedABC
from pydantic import Extra


class VisualizationWrapper(TracedABC):
    def __init__(self, og_obj, og_fn):
        self.og_obj = og_obj
        self.og_fn = og_fn

    async def run(self, *args, **kwargs):
        return self.og_fn(self.og_obj, *args, **kwargs)


class PydanticBypass:
    def __init__(self, obj):
        self.obj = obj
        self.has_config = hasattr(self.obj.__class__, "__config__")
        self.has_extra = self.has_config and hasattr(
            self.obj.__class__.__config__, "extra"
        )

    def __enter__(self):
        if self.has_extra:
            self.obj.__class__.__config__.extra = Extra.allow

    def __exit__(self, *args):
        if self.has_extra:
            self.obj.__class__.__config__.extra = Extra.forbid


def ice_agent_getter(og_fn):
    def get_ice_agent(self):
        # we have to do this roundabout thing because we don't know ahead of time all
        # the classes we want to set ice_agent for
        if not hasattr(self, "ice_agent"):
            with PydanticBypass(self):
                self.ice_agent = VisualizationWrapper(self, og_fn=og_fn)
        return self.ice_agent

    return get_ice_agent


def overridden_call(self, *args, **kwargs):
    """Preserve sync nature of OG call method"""
    ice_agent = self.get_ice_agent()
    if not hasattr(self.__class__, "_should_trace") or self.__class__._should_trace:
        # ICE displays class name in visualization
        ice_agent.__class__.__name__ = self.__class__.__name__
        return asyncio.get_event_loop().run_until_complete(
            ice_agent.run(*args, **kwargs)
        )

    return ice_agent.og_fn(self, *args, **kwargs)


def hijack(cls, fn_name, get_replacement):
    replacement = get_replacement(getattr(cls, fn_name))
    setattr(cls, fn_name, replacement)


def ice_hijack(cls, og_method_name):
    og_method = getattr(cls, og_method_name)
    setattr(cls, "get_ice_agent", ice_agent_getter(og_method))
    setattr(cls, og_method_name, overridden_call)
