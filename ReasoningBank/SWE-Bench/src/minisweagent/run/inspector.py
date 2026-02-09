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

#!/usr/bin/env python3
"""
Simple trajectory inspector for browsing agent conversation trajectories.

[not dim]
More information about the usage: [bold green]https://mini-swe-agent.com/latest/usage/inspector/[/bold green]
[/not dim]
"""

import json
import os
from pathlib import Path

import typer
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Static

from minisweagent.agents.interactive_textual import _messages_to_steps

app = typer.Typer(rich_markup_mode="rich", add_completion=False)


class TrajectoryInspector(App):
    BINDINGS = [
        Binding("right,l", "next_step", "Step++"),
        Binding("left,h", "previous_step", "Step--"),
        Binding("0", "first_step", "Step=0"),
        Binding("$", "last_step", "Step=-1"),
        Binding("j,down", "scroll_down", "Scroll down"),
        Binding("k,up", "scroll_up", "Scroll up"),
        Binding("L", "next_trajectory", "Next trajectory"),
        Binding("H", "previous_trajectory", "Previous trajectory"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, trajectory_files):
        css_path = os.environ.get(
            "MSWEA_INSPECTOR_STYLE_PATH", str(Path(__file__).parent.parent / "config" / "mini.tcss")
        )
        self.__class__.CSS = Path(css_path).read_text()

        super().__init__()
        self.trajectory_files = trajectory_files
        self._i_trajectory = 0
        self._i_step = 0
        self.messages = []
        self.steps = []

        if trajectory_files:
            self._load_current_trajectory()

    # --- Basics ---

    @property
    def i_step(self):
        """Current step index."""
        return self._i_step

    @i_step.setter
    def i_step(self, value):
        """Set current step index, automatically clamping to valid bounds."""
        if value != self._i_step and self.n_steps > 0:
            self._i_step = max(0, min(value, self.n_steps - 1))
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_steps(self):
        """Number of steps in current trajectory."""
        return len(self.steps)

    @property
    def i_trajectory(self):
        """Current trajectory index."""
        return self._i_trajectory

    @i_trajectory.setter
    def i_trajectory(self, value):
        """Set current trajectory index, automatically clamping to valid bounds."""
        if value != self._i_trajectory and self.n_trajectories > 0:
            self._i_trajectory = max(0, min(value, self.n_trajectories - 1))
            self._load_current_trajectory()
            self.query_one(VerticalScroll).scroll_to(y=0, animate=False)
            self.update_content()

    @property
    def n_trajectories(self):
        """Number of trajectory files."""
        return len(self.trajectory_files)

    def _load_current_trajectory(self):
        """Load the currently selected trajectory file."""
        if not self.trajectory_files:
            self.messages = []
            self.steps = []
            return

        trajectory_file = self.trajectory_files[self.i_trajectory]
        try:
            data = json.loads(trajectory_file.read_text())

            if isinstance(data, list):
                self.messages = data
            elif isinstance(data, dict) and "messages" in data:
                self.messages = data["messages"]
            else:
                raise ValueError("Unrecognized trajectory format")

            self.steps = _messages_to_steps(self.messages)
            self._i_step = 0
        except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
            self.messages = []
            self.steps = []
            self.notify(f"Error loading {trajectory_file.name}: {e}", severity="error")

    @property
    def current_trajectory_name(self):
        """Get the name of the current trajectory file."""
        if not self.trajectory_files:
            return "No trajectories"
        return self.trajectory_files[self.i_trajectory].name

    def compose(self):
        yield Header()
        with Container(id="main"):
            with VerticalScroll():
                yield Vertical(id="content")
        yield Footer()

    def on_mount(self):
        self.update_content()

    def update_content(self):
        """Update the displayed content."""
        container = self.query_one("#content", Vertical)
        container.remove_children()

        if not self.steps:
            container.mount(Static("No trajectory loaded or empty trajectory"))
            self.title = "Trajectory Inspector - No Data"
            return

        for message in self.steps[self.i_step]:
            if isinstance(message["content"], list):
                content_str = "\n".join([item["text"] for item in message["content"]])
            else:
                content_str = str(message["content"])
            message_container = Vertical(classes="message-container")
            container.mount(message_container)
            role = message["role"].replace("assistant", "mini-swe-agent")
            message_container.mount(Static(role.upper(), classes="message-header"))
            message_container.mount(Static(Text(content_str, no_wrap=False), classes="message-content"))

        self.title = (
            f"Trajectory {self.i_trajectory + 1}/{self.n_trajectories} - "
            f"{self.current_trajectory_name} - "
            f"Step {self.i_step + 1}/{self.n_steps}"
        )

    # --- Navigation actions ---

    def action_next_step(self):
        self.i_step += 1

    def action_previous_step(self):
        self.i_step -= 1

    def action_first_step(self):
        self.i_step = 0

    def action_last_step(self):
        self.i_step = self.n_steps - 1

    def action_next_trajectory(self):
        self.i_trajectory += 1

    def action_previous_trajectory(self):
        self.i_trajectory -= 1

    def action_scroll_down(self):
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y + 15)

    def action_scroll_up(self):
        vs = self.query_one(VerticalScroll)
        vs.scroll_to(y=vs.scroll_target_y - 15)


@app.command(help=__doc__)
def main(
    path = typer.Argument(".", help="Directory to search for trajectory files or specific trajectory file"),
):
    path_obj = Path(path)

    if path_obj.is_file():
        trajectory_files = [path_obj]
    elif path_obj.is_dir():
        trajectory_files = sorted(path_obj.rglob("*.traj.json"))
        if not trajectory_files:
            raise typer.BadParameter(f"No trajectory files found in '{path}'")
    else:
        raise typer.BadParameter(f"Error: Path '{path}' does not exist")

    inspector = TrajectoryInspector(trajectory_files)
    inspector.run()


if __name__ == "__main__":
    app()
