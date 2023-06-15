import asyncio
import inspect

import gorilla
from ice.trace import TracedABC

LANGCHAIN_VISUALIZER_PATCH_ID = "lc-viz"
# override prefix used by vcr-langchain for visualization compatibility
VCR_VIZ_INTEROP_PREFIX = "_vcr_"


class VisualizationWrapper(TracedABC):
    def __init__(self, og_obj, og_fn):
        self.og_obj = og_obj
        self.og_fn = og_fn

    @property
    def is_async(self):
        return inspect.iscoroutinefunction(self.og_fn)

    async def run(self, *args, **kwargs):
        # Async function that gets visualized.
        #
        # Invocation of this function is what triggers visibility in the ICE execution
        # graph. Override this function if you want args and kwargs to be visualized as
        # actual named arguments rather than as arrays and dicts.
        #
        # The docstring gets visualized too, which is why this documentation is not in
        # the docstring.
        if self.is_async:
            return await self.og_fn(self.og_obj, *args, **kwargs)
        else:
            return self.og_fn(self.og_obj, *args, **kwargs)


def get_viz_wrapper(viz_cls, og_self, og_fn_name: str):
    """
    Return the visualization wrapper object.

    This object's "run" function will be what triggers ICE execution capture.
    """
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
    """
    Get the new function that will override the original function.

    The returned function will end up calling the original function, but in a way that
    causes the call to be recorded by ICE.
    """

    def overridden_call(og_self, *args, **kwargs):
        ice_agent = get_viz_wrapper(viz_cls, og_self, og_method_name)
        if (
            not hasattr(og_self.__class__, "_should_trace")
            or og_self.__class__._should_trace
        ):
            # ICE displays class name in visualization
            ice_agent.__class__.__name__ = og_self.__class__.__name__
            # this is not the original class's "run" function -- in fact, the original
            # function can be named anything, since the name is stored in
            # og_method_name. Instead, this is the visualization wrapper's "run"
            # function, which is what gets visualized.
            return asyncio.get_event_loop().run_until_complete(
                ice_agent.run(*args, **kwargs)
            )

        return ice_agent.og_fn(og_self, *args, **kwargs)

    return overridden_call


def get_async_overridden_call(viz_cls, og_method_name):
    """
    Like get_overridden_call, but returns an async override.
    """

    async def overridden_call(og_self, *args, **kwargs):
        ice_agent = get_viz_wrapper(viz_cls, og_self, og_method_name)
        if (
            not hasattr(og_self.__class__, "_should_trace")
            or og_self.__class__._should_trace
        ):
            # ICE displays class name in visualization
            ice_agent.__class__.__name__ = og_self.__class__.__name__
            # this is not the original class's "run" function -- in fact, the original
            # function can be named anything, since the name is stored in
            # og_method_name. Instead, this is the visualization wrapper's "run"
            # function, which is what gets visualized.
            return await ice_agent.run(*args, **kwargs)

        return await ice_agent.og_fn(og_self, *args, **kwargs)

    return overridden_call


def hijack(cls, fn_name, get_replacement):
    replacement = get_replacement(getattr(cls, fn_name))
    setattr(cls, fn_name, replacement)


def ice_hijack(cls, og_method_name, viz_cls=VisualizationWrapper):
    """
    Hijack cls.og_method_name to refer to an overridden call.

    The overridden call will have a chance to call the original function.
    """
    og_fn = getattr(cls, og_method_name)
    is_async = inspect.iscoroutinefunction(og_fn)
    overridden_call = (
        get_async_overridden_call(viz_cls, og_method_name)
        if is_async
        else get_overridden_call(viz_cls, og_method_name)
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
