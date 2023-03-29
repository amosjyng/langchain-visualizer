import asyncio
import sys
from functools import wraps
from inspect import iscoroutinefunction
from traceback import print_exc

from ice.environment import env
from ice.mode import Mode
from ice.recipe import FunctionBasedRecipe, recipe
from ice.trace import enable_trace, trace
from merge_args import merge_args


def visualize(fn: FunctionBasedRecipe):
    def new_main(self, main: FunctionBasedRecipe):
        if not iscoroutinefunction(main):
            raise TypeError("visualize must be given an async function")

        # Trace all globals defined in main's module.
        try:
            g = main.__globals__
        except AttributeError:
            # Perhaps this is a functools.partial
            g = main.func.__globals__  # type: ignore[attr-defined]
        for name, value in g.items():
            if getattr(value, "__module__", None) == main.__module__:
                g[name] = trace(value)

        traced_main = trace(main)
        self.all_recipes.append(traced_main)

        # The frontend shows everything under the first traced root.
        # TODO: Once main.py is gone, change the frontend and get rid of this wrapper.
        @trace
        @wraps(main)
        async def hidden_wrapper(*args, **kwargs):
            try:
                result = await traced_main(*args, **kwargs)
            except NameError:
                print_exc()
                print(
                    "\nReminder: recipe.main should be at the bottom of the file",
                    file=sys.stderr,
                )
                sys.exit(1)

            env().print(result, format_markdown=False)
            return result

        # A traced function cannot be called until the event loop is running.
        @wraps(main)
        async def untraced_wrapper(*args, **kwargs):
            return await hidden_wrapper(*args, **kwargs)

        @merge_args(main)
        def cli(
            *args,
            mode: Mode = "machine",
            trace: bool = True,
            **kwargs,
        ):
            self._mode = mode
            if trace:
                enable_trace()
            asyncio.run(untraced_wrapper(*args, **kwargs))

        cli()

    return new_main(recipe, fn)
