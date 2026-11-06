from __future__ import annotations

from nicegui import ui

__all__ = ["ThreeWaySplitter"]


class ThreeWaySplitter:
    """Reusable three-way splitter with left, middle, and right content slots."""

    def __init__(self, *, left_value: float = 20, right_value: float = 20) -> None:
        self._left_splitter_prev_value = left_value
        self._right_splitter_prev_value = right_value

        self.left_splitter = ui.splitter(value=left_value).classes("w-full h-full")
        with self.left_splitter.after:
            self.right_splitter = ui.splitter(value=right_value, reverse=True).classes(
                "w-full h-full"
            )

        self._add_splitter_toggle(
            self.left_splitter,
            prev_value_attr="_left_splitter_prev_value",
            open_icon="keyboard_double_arrow_left",
            collapsed_icon="keyboard_double_arrow_right",
        )
        self._add_splitter_toggle(
            self.right_splitter,
            prev_value_attr="_right_splitter_prev_value",
            open_icon="keyboard_double_arrow_right",
            collapsed_icon="keyboard_double_arrow_left",
        )

    @staticmethod
    def _splitter_icon_name(value: float, open_icon: str, collapsed_icon: str) -> str:
        return collapsed_icon if value < 1 else open_icon

    def _add_splitter_toggle(
        self,
        splitter,
        *,
        prev_value_attr: str,
        open_icon: str,
        collapsed_icon: str,
    ) -> None:
        with splitter.separator:
            with (
                ui
                .button(color="secondary")
                .classes("p-0")
                .props("round")
                .on_click(lambda: self._toggle_splitter(splitter, prev_value_attr))
            ):
                icon = ui.icon(
                    self._splitter_icon_name(splitter.value, open_icon, collapsed_icon)
                )

        splitter.on_value_change(
            lambda e, icon=icon: icon.set_name(
                self._splitter_icon_name(e.value, open_icon, collapsed_icon)
            )
        )

    def _toggle_splitter(self, splitter, prev_value_attr: str) -> None:
        if splitter.value < 1:
            splitter.value = getattr(self, prev_value_attr)
        else:
            setattr(self, prev_value_attr, splitter.value)
            splitter.value = 0

    @property
    def left(self):
        return self.left_splitter.before

    @property
    def middle(self):
        return self.right_splitter.before

    @property
    def right(self):
        return self.right_splitter.after
