import logging
import os
import shutil
from typing import Generator

import pytest
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.webdriver import WebDriver

from nicegui.testing import Screen

os.environ.setdefault("MPLBACKEND", "Agg")

pytest_plugins = ["nicegui.testing.plugin"]

if Screen is not None:
    Screen.SCREENSHOT_PASSED_TESTS = False


# This can be removed once https://github.com/zauberzeug/nicegui/pull/6484 is merged and released.
@pytest.fixture
def nicegui_driver(nicegui_chrome_options: Options) -> Generator[WebDriver, None, None]:
    """Create a new Chrome driver instance (reused across tests in the session)."""
    for executable_path in (
        # Required for Ubuntu Snaps:
        "/snap/bin/chromium.chromedriver",
        None,
        # Required for ARM devcontainers:
        shutil.which("chromedriver"),
        "chromedriver",
    ):
        logging.info(
            "Trying to start Chrome WebDriver with executable_path=%s", executable_path
        )
        try:
            s = Service(executable_path=executable_path)
            driver_ = WebDriver(service=s, options=nicegui_chrome_options)
            break
        except Exception:
            continue
    else:  # no break
        raise RuntimeError("Could not start Chrome WebDriver.")
    driver_.implicitly_wait(Screen.IMPLICIT_WAIT)
    driver_.set_page_load_timeout(4)
    yield driver_
    driver_.quit()
