import threading

from ice.logging import log_lock
from ice.server import ensure_server_running, is_server_running
from ice.settings import settings
from ice.trace import Trace
from IPython.display import IFrame

from .visualize import visualize as regular_visualize

latest_viz_url = None
evt = threading.Event()


def new_server_and_browser(self):
    global evt
    global latest_viz_url
    # We use this lock to prevent logging from here (which runs in a
    # background thread) from burying the input prompt in
    # [Settings.__get_and_store].
    with log_lock:
        is_running = None
        if settings.OUGHT_ICE_AUTO_SERVER:
            ensure_server_running()
            is_running = True

        if not settings.OUGHT_ICE_AUTO_BROWSER:
            return

        is_running = is_running or is_server_running()
        if not is_running:
            return

        latest_viz_url = self.url
        evt.set()


Trace._server_and_browser = new_server_and_browser  # type: ignore


def visualize(fn, width: int = 1000, height: int = 500):
    threading.Thread(target=regular_visualize, args=(fn,)).start()
    evt.wait(timeout=20)
    print(f"Rendering {latest_viz_url} in notebook")
    return IFrame(latest_viz_url, width=width, height=height)
