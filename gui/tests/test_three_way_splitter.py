from __future__ import annotations

from selenium.webdriver.common.by import By

from nicegui import ui
from nicegui.testing import Screen

from polytaxo_gui.components import ThreeWaySplitter


def test_three_way_splitter_renders_all_slots(screen: Screen):
    @ui.page("/")
    def page():
        splitter = ThreeWaySplitter()
        with splitter.left:
            ui.label("Left panel")
        with splitter.middle:
            ui.label("Middle panel")
        with splitter.right:
            ui.label("Right panel")

    screen.open("/")
    screen.should_contain("Left panel")
    screen.should_contain("Middle panel")
    screen.should_contain("Right panel")


def test_three_way_splitter_toggles_and_restores_previous_values(screen: Screen):
    splitter = None

    @ui.page("/")
    def page():
        nonlocal splitter
        splitter = ThreeWaySplitter()

    screen.open("/")

    assert splitter is not None
    screen.wait_for(lambda: splitter.left_splitter.value == 20)
    screen.wait_for(lambda: splitter.right_splitter.value == 20)

    toggle_buttons = screen.selenium.find_elements(
        By.CSS_SELECTOR, ".q-splitter__separator .q-btn"
    )
    assert len(toggle_buttons) == 2

    screen.selenium.execute_script("arguments[0].click();", toggle_buttons[0])
    screen.wait_for(lambda: splitter.left_splitter.value == 0)

    toggle_buttons = screen.selenium.find_elements(
        By.CSS_SELECTOR, ".q-splitter__separator .q-btn"
    )
    screen.selenium.execute_script("arguments[0].click();", toggle_buttons[0])
    screen.wait_for(lambda: splitter.left_splitter.value == 20)

    toggle_buttons = screen.selenium.find_elements(
        By.CSS_SELECTOR, ".q-splitter__separator .q-btn"
    )
    screen.selenium.execute_script("arguments[0].click();", toggle_buttons[1])
    screen.wait_for(lambda: splitter.right_splitter.value == 0)

    toggle_buttons = screen.selenium.find_elements(
        By.CSS_SELECTOR, ".q-splitter__separator .q-btn"
    )
    screen.selenium.execute_script("arguments[0].click();", toggle_buttons[1])
    screen.wait_for(lambda: splitter.right_splitter.value == 20)
