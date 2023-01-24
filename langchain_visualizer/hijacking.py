import asyncio

import gorilla
from ice.trace import TracedABC
from pydantic import Extra

LANGCHAIN_VISUALIZER_PATCH_ID = "lc-viz"
VCR_LANGCHAIN_PATCH_ID = "lc-vcr"
# override prefix used by vcr-langchain for visualization compatibility
VCR_VIZ_INTEROP_PREFIX = "_vcr_"


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


def ice_agent_getter(og_fn_name: str):
    def get_ice_agent(self):
        # we have to do this roundabout thing because we don't know ahead of time all
        # the classes we want to set ice_agent for
        if not hasattr(self, "ice_agent"):
            with PydanticBypass(self):
                vcr_key = VCR_VIZ_INTEROP_PREFIX + og_fn_name
                if hasattr(self.__class__, vcr_key):
                    # if vcr-langchain is here as well, call them so that caching can
                    # happen
                    og_fn = getattr(self.__class__, vcr_key)
                else:
                    og_fn = gorilla.get_original_attribute(
                        self.__class__, og_fn_name, LANGCHAIN_VISUALIZER_PATCH_ID
                    )
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
    gorilla.apply(
        gorilla.Patch(
            destination=cls,
            name="get_ice_agent",
            obj=ice_agent_getter(og_method_name),
            settings=gorilla.Settings(allow_hit=False),
        )
    )
    gorilla.apply(
        gorilla.Patch(
            destination=cls,
            name=og_method_name,
            obj=overridden_call,
            settings=gorilla.Settings(store_hit=True, allow_hit=True),
        ),
        id=LANGCHAIN_VISUALIZER_PATCH_ID,
    )
