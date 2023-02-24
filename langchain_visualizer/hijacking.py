import asyncio

import gorilla
from ice.trace import TracedABC

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


def get_viz_wrapper(viz_cls, og_self, og_fn_name: str):
    vcr_key = VCR_VIZ_INTEROP_PREFIX + og_fn_name
    if hasattr(og_self.__class__, vcr_key):
        # if vcr-langchain is here as well, call them so that caching can
        # happen
        og_fn = getattr(og_self.__class__, vcr_key)
    else:
        og_fn = gorilla.get_original_attribute(
            og_self.__class__, og_fn_name, LANGCHAIN_VISUALIZER_PATCH_ID
        )
    return viz_cls(og_self, og_fn=og_fn)


def get_overridden_call(viz_cls, og_method_name):
    def overridden_call(og_self, *args, **kwargs):
        """Preserve sync nature of OG call method"""
        ice_agent = get_viz_wrapper(viz_cls, og_self, og_method_name)
        if (
            not hasattr(og_self.__class__, "_should_trace")
            or og_self.__class__._should_trace
        ):
            # ICE displays class name in visualization
            ice_agent.__class__.__name__ = og_self.__class__.__name__
            return asyncio.get_event_loop().run_until_complete(
                ice_agent.run(*args, **kwargs)
            )

        return ice_agent.og_fn(og_self, *args, **kwargs)

    return overridden_call


def hijack(cls, fn_name, get_replacement):
    replacement = get_replacement(getattr(cls, fn_name))
    setattr(cls, fn_name, replacement)


def ice_hijack(cls, og_method_name, viz_cls=VisualizationWrapper):
    gorilla.apply(
        gorilla.Patch(
            destination=cls,
            name=og_method_name,
            obj=get_overridden_call(viz_cls, og_method_name),
            settings=gorilla.Settings(store_hit=True, allow_hit=True),
        ),
        id=LANGCHAIN_VISUALIZER_PATCH_ID,
    )
