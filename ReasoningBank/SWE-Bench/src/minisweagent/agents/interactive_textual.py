# coding=utf-8
# Copyright 2026 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Extension of the `default.py` agent that uses Textual for an interactive TUI.
For a simpler version of an interactive UI that does not require threading and more, see `interactive.py`.
"""

import logging
import os
import re
import threading
import time
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rich.spinner import Spinner
from rich.text import Text
from textual.app import App, ComposeResult, SystemCommand
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import Screen
from textual.widgets import Footer, Header, Input, Static, TextArea

from minisweagent.agents.default import AgentConfig, DefaultAgent, NonTerminatingException, Submitted


@dataclass
class TextualAgentConfig(AgentConfig):
    mode: Literal["confirm", "yolo"] = "confirm"
    """Mode for action execution: 'confirm' requires user confirmation, 'yolo' executes immediately."""
    whitelist_actions: list[str] = field(default_factory=list)
    """Never confirm actions that match these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""


class _TextualAgent(DefaultAgent):
    def __init__(self, app, *args, **kwargs):
        """Connects the DefaultAgent to the TextualApp."""
        self.app = app
        super().__init__(*args, config_class=TextualAgentConfig, **kwargs)
        self._current_action_from_human = False

    def add_message(self, role, content, **kwargs):
        super().add_message(role, content, **kwargs)
        if self.app.agent_state != "UNINITIALIZED":
            self.app.call_from_thread(self.app.on_message_added)

    def query(self):
        if self.config.mode == "human":
            human_input = self.app.input_container.request_input("Enter your command:")
            self._current_action_from_human = True
            msg = {"content": f"\n```bash\n{human_input}\n```"}
            self.add_message("assistant", msg["content"])
            return msg
        self._current_action_from_human = False
        return super().query()

    def run(self, task, **kwargs):
        try:
            exit_status, result = super().run(task, **kwargs)
        except Exception as e:
            result = str(e)
            self.app.call_from_thread(self.app.action_quit)
            print(traceback.format_exc())
            return "ERROR", result
        else:
            self.app.call_from_thread(self.app.on_agent_finished, exit_status, result)
        self.app.call_from_thread(self.app.action_quit)
        return exit_status, result

    def execute_action(self, action):
        if self.config.mode == "human" and not self._current_action_from_human:  # threading, grrrrr
            raise NonTerminatingException("Command not executed because user switched to manual mode.")
        if (
            self.config.mode == "confirm"
            and action["action"].strip()
            and not any(re.match(r, action["action"]) for r in self.config.whitelist_actions)
        ):
            result = self.app.input_container.request_input("Press ENTER to confirm or provide rejection reason")
            if result:  # Non-empty string means rejection
                raise NonTerminatingException(f"Command not executed: {result}")
        return super().execute_action(action)

    def has_finished(self, output):
        try:
            return super().has_finished(output)
        except Submitted as e:
            if self.config.confirm_exit:
                if new_task := self.app.input_container.request_input(
                    "[bold green]Agent wants to finish.[/bold green] "
                    "[green]Type a comment to give it a new task or press enter to quit.\n"
                ).strip():
                    raise NonTerminatingException(f"The user added a new task: {new_task}")
            raise e


class AddLogEmitCallback(logging.Handler):
    def __init__(self, callback):
        """Custom log handler that forwards messages via callback."""
        super().__init__()
        self.callback = callback

    def emit(self, record):
        self.callback(record)  # type: ignore[attr-defined]


def _messages_to_steps(messages):
    """Group messages into "pages" as shown by the UI."""
    steps = []
    current_step = []
    for message in messages:
        current_step.append(message)
        if message["role"] == "user":
            steps.append(current_step)
            current_step = []
    if current_step:
        steps.append(current_step)
    return steps


class SmartInputContainer(Container):
    def __init__(self, app):
        """Smart input container supporting single-line and multi-line input modes."""
        super().__init__(classes="smart-input-container")
        self._app = app
        self._multiline_mode = False
        self.can_focus = True
        self.display = False

        self.pending_prompt: str | None = None
        self._input_event = threading.Event()
        self._input_result: str | None = None

        self._header_display = Static(id="input-header-display", classes="message-header input-request-header")
        self._hint_text = Static(classes="hint-text")
        self._single_input = Input(placeholder="Type your input...")
        self._multi_input = TextArea(show_line_numbers=False, classes="multi-input")
        self._input_elements_container = Vertical(
            self._header_display,
            self._hint_text,
            self._single_input,
            self._multi_input,
            classes="message-container",
        )

    def compose(self):
        yield self._input_elements_container

    def on_mount(self):
        """Initialize the widget state."""
        self._multi_input.display = False
        self._update_mode_display()

    def on_focus(self):
        """Called when the container gains focus."""
        if self._multiline_mode:
            self._multi_input.focus()
        else:
            self._single_input.focus()

    def request_input(self, prompt):
        """Request input from user. Returns input text (empty string if confirmed without reason)."""
        self._input_event.clear()
        self._input_result = None
        self.pending_prompt = prompt
        self._header_display.update(prompt)
        self._update_mode_display()
        self._app.call_from_thread(self._app.update_content)
        self._input_event.wait()
        return self._input_result or ""

    def _complete_input(self, input_text):
        """Internal method to complete the input process."""
        self._input_result = input_text
        self.pending_prompt = None
        self.display = False
        self._single_input.value = ""
        self._multi_input.text = ""
        self._multiline_mode = False
        self._update_mode_display()
        self._app.agent_state = "RUNNING"
        self._app.update_content()
        # Reset scroll position to bottom since input container disappearing changes layout
        # somehow scroll_to doesn't work.
        self._app._vscroll.scroll_y = 0
        self._input_event.set()

    def action_toggle_mode(self):
        """Switch from single-line to multi-line mode (one-way only)."""
        if self.pending_prompt is None or self._multiline_mode:
            return

        self._multiline_mode = True
        self._update_mode_display()
        self.on_focus()

    def _update_mode_display(self):
        """Update the display based on current mode."""
        if self._multiline_mode:
            self._multi_input.text = self._single_input.value
            self._single_input.display = False
            self._multi_input.display = True
            self._hint_text.update(
                "[reverse][bold][$accent] Ctrl+D [/][/][/] to submit, [reverse][bold][$accent] Tab [/][/][/] to switch focus with other controls"
            )
        else:
            self._hint_text.update(
                "[reverse][bold][$accent] Enter [/][/][/] to submit, [reverse][bold][$accent] Ctrl+T [/][/][/] to switch to multi-line input, [reverse][bold][$accent] Tab [/][/][/] to switch focus with other controls",
            )
            self._multi_input.display = False
            self._single_input.display = True

    def on_input_submitted(self, event):
        """Handle single-line input submission."""
        if not self._multiline_mode:
            text = event.input.value.strip()
            self._complete_input(text)

    def on_key(self, event):
        """Handle key events."""
        if event.key == "ctrl+t" and not self._multiline_mode:
            event.prevent_default()
            self.action_toggle_mode()
            return

        if self._multiline_mode and event.key == "ctrl+d":
            event.prevent_default()
            self._complete_input(self._multi_input.text.strip())
            return

        if event.key == "escape":
            event.prevent_default()
            self.can_focus = False
            self._app.set_focus(None)
            return


class TextualAgent(App):
    BINDINGS = [
        Binding("right,l", "next_step", "Step++", tooltip="Show next step of the agent"),
        Binding("left,h", "previous_step", "Step--", tooltip="Show previous step of the agent"),
        Binding("0", "first_step", "Step=0", tooltip="Show first step of the agent", show=False),
        Binding("$", "last_step", "Step=-1", tooltip="Show last step of the agent", show=False),
        Binding("j,down", "scroll_down", "Scroll down", show=False),
        Binding("k,up", "scroll_up", "Scroll up", show=False),
        Binding("q,ctrl+q", "quit", "Quit", tooltip="Quit the agent"),
        Binding("y,ctrl+y", "yolo", "YOLO mode", tooltip="Switch to YOLO Mode (LM actions will execute immediately)"),
        Binding(
            "c",
            "confirm",
            "CONFIRM mode",
            tooltip="Switch to Confirm Mode (LM proposes commands and you confirm/reject them)",
        ),
        Binding("u,ctrl+u", "human", "HUMAN mode", tooltip="Switch to Human Mode (you can now type commands directly)"),
        Binding("f1,question_mark", "toggle_help_panel", "Help", tooltip="Show help"),
    ]

    def __init__(self, model, env, **kwargs):
        css_path = os.environ.get("MSWEA_MINI_STYLE_PATH", str(Path(__file__).parent.parent / "config" / "mini.tcss"))
        self.__class__.CSS = Path(css_path).read_text()
        super().__init__()
        self.agent_state = "UNINITIALIZED"
        self.agent = _TextualAgent(self, model=model, env=env, **kwargs)
        self._i_step = 0
        self.n_steps = 1
        self.input_container = SmartInputContainer(self)
        self.log_handler = AddLogEmitCallback(lambda record: self.call_from_thread(self.on_log_message_emitted, record))
        logging.getLogger().addHandler(self.log_handler)
        self._spinner = Spinner("dots")
        self.exit_status: str = "ExitStatusUnset"
        self.result: str = ""

        self._vscroll = VerticalScroll()

    def run(self, task, **kwargs):
        threading.Thread(target=lambda: self.agent.run(task, **kwargs), daemon=True).start()
        super().run()
        return self.exit_status, self.result

    # --- Basics ---

    @property
    def config(self):
        return self.agent.config

    @property
    def i_step(self):
        """Current step index."""
        return self._i_step

    @i_step.setter
    def i_step(self, value):
        """Set current step index, automatically clamping to valid bounds."""
        if value != self._i_step:
            self._i_step = max(0, min(value, self.n_steps - 1))
            self._vscroll.scroll_to(y=0, animate=False)
            self.update_content()

    def compose(self):
        yield Header()
        with Container(id="main"):
            with self._vscroll:
                with Vertical(id="content"):
                    pass
                yield self.input_container
        yield Footer()

    def on_mount(self):
        self.agent_state = "RUNNING"
        self.update_content()
        self.set_interval(1 / 8, self._update_headers)

    @property
    def messages(self):
        return self.agent.messages

    @property
    def model(self):
        return self.agent.model

    @property
    def env(self):
        return self.agent.env

    # --- Reacting to events ---

    def on_message_added(self):
        auto_follow = self.i_step == self.n_steps - 1 and self._vscroll.scroll_y <= 1
        self.n_steps = len(_messages_to_steps(self.agent.messages))
        self.update_content()
        if auto_follow:
            self.action_last_step()

    def on_log_message_emitted(self, record):
        """Handle log messages of warning level or higher by showing them as notifications."""
        if record.levelno >= logging.WARNING:
            self.notify(f"[{record.levelname}] {record.getMessage()}", severity="warning")

    def on_unmount(self):
        """Clean up the log handler when the app shuts down."""
        if hasattr(self, "log_handler"):
            logging.getLogger().removeHandler(self.log_handler)

    def on_agent_finished(self, exit_status, result):
        self.agent_state = "STOPPED"
        self.notify(f"Agent finished with status: {exit_status}")
        self.exit_status = exit_status
        self.result = result
        self.update_content()

    # --- UI update logic ---

    def update_content(self):
        container = self.query_one("#content", Vertical)
        container.remove_children()
        items = _messages_to_steps(self.agent.messages)

        if not items:
            container.mount(Static("Waiting for agent to start..."))
            return

        for message in items[self.i_step]:
            if isinstance(message["content"], list):
                content_str = "\n".join([item["text"] for item in message["content"]])
            else:
                content_str = str(message["content"])
            message_container = Vertical(classes="message-container")
            container.mount(message_container)
            role = message["role"].replace("assistant", "mini-swe-agent")
            message_container.mount(Static(role.upper(), classes="message-header"))
            message_container.mount(Static(Text(content_str, no_wrap=False), classes="message-content"))

        if self.input_container.pending_prompt is not None:
            self.agent_state = "AWAITING_INPUT"
        self.input_container.display = self.input_container.pending_prompt is not None and self.i_step == len(items) - 1
        if self.input_container.display:
            self.input_container.on_focus()

        self._update_headers()
        self.refresh()

    def _update_headers(self):
        """Update just the title with current state and spinner if needed."""
        status_text = self.agent_state
        if self.agent_state == "RUNNING":
            spinner_frame = str(self._spinner.render(time.time())).strip()
            status_text = f"{self.agent_state} {spinner_frame}"
        self.title = f"Step {self.i_step + 1}/{self.n_steps} - {status_text} - Cost: ${self.agent.model.cost:.2f}"
        try:
            self.query_one("Header").set_class(self.agent_state == "RUNNING", "running")
        except NoMatches:  # might be called when shutting down
            pass

    # --- Other textual overrides ---

    def get_system_commands(self, screen):
        # Add to palette
        yield from super().get_system_commands(screen)
        for binding in self.BINDINGS:
            description = f"{binding.description} (shortcut {' OR '.join(binding.key.split(','))})"  # type: ignore[attr-defined]
            yield SystemCommand(description, binding.tooltip, binding.action)  # type: ignore[attr-defined]

    # --- Textual bindings ---

    def action_yolo(self):
        self.agent.config.mode = "yolo"
        if self.input_container.pending_prompt is not None:
            self.input_container._complete_input("")  # accept
        self.notify("YOLO mode enabled - LM actions will execute immediately")

    def action_human(self):
        if self.agent.config.mode == "confirm" and self.input_container.pending_prompt is not None:
            self.input_container._complete_input("User switched to manual mode, this command will be ignored")
        self.agent.config.mode = "human"
        self.notify("Human mode enabled - you can now type commands directly")

    def action_confirm(self):
        if self.agent.config.mode == "human" and self.input_container.pending_prompt is not None:
            self.input_container._complete_input("")  # just submit blank action
        self.agent.config.mode = "confirm"
        self.notify("Confirm mode enabled - LM proposes commands and you confirm/reject them")

    def action_next_step(self):
        self.i_step += 1

    def action_previous_step(self):
        self.i_step -= 1

    def action_first_step(self):
        self.i_step = 0

    def action_last_step(self):
        self.i_step = self.n_steps - 1

    def action_scroll_down(self):
        self._vscroll.scroll_to(y=self._vscroll.scroll_target_y + 15)

    def action_scroll_up(self):
        self._vscroll.scroll_to(y=self._vscroll.scroll_target_y - 15)

    def action_toggle_help_panel(self):
        if self.query("HelpPanel"):
            self.action_hide_help_panel()
        else:
            self.action_show_help_panel()
