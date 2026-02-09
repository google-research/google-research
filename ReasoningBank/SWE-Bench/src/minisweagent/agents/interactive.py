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

"""A small generalization of the default agent that puts the user in the loop.

There are three modes:
- human: commands issued by the user are executed immediately
- confirm: commands issued by the LM but not whitelisted are confirmed by the user
- yolo: commands issued by the LM are executed immediately without confirmation
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from rich.rule import Rule

from minisweagent import global_config_dir
from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, Submitted

console = Console(highlight=False)
prompt_session = PromptSession(history=FileHistory(global_config_dir / "interactive_history.txt"))


@dataclass
class InteractiveAgentConfig(AgentConfig):
    mode: Literal["human", "confirm", "yolo"] = "confirm"
    """Whether to confirm actions."""
    whitelist_actions: list[str] = field(default_factory=list)
    """Never confirm actions that match these regular expressions."""
    confirm_exit: bool = True
    """If the agent wants to finish, do we ask for confirmation from user?"""


class InteractiveAgent(DefaultAgent):
    _MODE_COMMANDS_MAPPING = {"/u": "human", "/c": "confirm", "/y": "yolo"}

    def __init__(self, *args, config_class=InteractiveAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.cost_last_confirmed = 0.0

    def add_message(self, role: str, content: str, **kwargs):
        # Extend supermethod to print messages
        super().add_message(role, content, **kwargs)
        if role == "assistant":
            console.print(
                f"\n[red][bold]mini-swe-agent[/bold] (step [bold]{self.model.n_calls}[/bold], [bold]${self.model.cost:.2f}[/bold]):[/red]\n",
                end="",
                highlight=False,
            )
        else:
            console.print(f"\n[bold green]{role.capitalize()}[/bold green]:\n", end="", highlight=False)
        console.print(content, highlight=False, markup=False)

    def query(self) -> dict:
        # Extend supermethod to handle human mode
        if self.config.mode == "human":
            match command := self._prompt_and_handle_special("[bold yellow]>[/bold yellow] "):
                case "/y" | "/c":  # Just go to the super query, which queries the LM for the next action
                    pass
                case _:
                    msg = {"content": f"\n```bash\n{command}\n```"}
                    self.add_message("assistant", msg["content"])
                    return msg
        try:
            with console.status("Waiting for the LM to respond..."):
                return super().query()
        except LimitsExceeded:
            console.print(
                f"Limits exceeded. Limits: {self.config.step_limit} steps, ${self.config.cost_limit}.\n"
                f"Current spend: {self.model.n_calls} steps, ${self.model.cost:.2f}."
            )
            self.config.step_limit = int(input("New step limit: "))
            self.config.cost_limit = float(input("New cost limit: "))
            return super().query()

    def step(self) -> dict:
        # Override the step method to handle user interruption
        try:
            console.print(Rule())
            return super().step()
        except KeyboardInterrupt:
            # We always add a message about the interrupt and then just proceed to the next step
            interruption_message = self._prompt_and_handle_special(
                "\n\n[bold yellow]Interrupted.[/bold yellow] "
                "[green]Type a comment/command[/green] (/h for available commands)"
                "\n[bold yellow]>[/bold yellow] "
            ).strip()
            if not interruption_message or interruption_message in self._MODE_COMMANDS_MAPPING:
                interruption_message = "Temporary interruption caught."
            raise NonTerminatingException(f"Interrupted by user: {interruption_message}")

    def execute_action(self, action: dict) -> dict:
        # Override the execute_action method to handle user confirmation
        if self.should_ask_confirmation(action["action"]):
            self.ask_confirmation()
        return super().execute_action(action)

    def should_ask_confirmation(self, action: str) -> bool:
        return self.config.mode == "confirm" and not any(re.match(r, action) for r in self.config.whitelist_actions)

    def ask_confirmation(self) -> None:
        prompt = (
            "[bold yellow]Execute?[/bold yellow] [green][bold]Enter[/bold] to confirm[/green], "
            "or [green]Type a comment/command[/green] (/h for available commands)\n"
            "[bold yellow]>[/bold yellow] "
        )
        match user_input := self._prompt_and_handle_special(prompt).strip():
            case "" | "/y":
                pass  # confirmed, do nothing
            case "/u":  # Skip execution action and get back to query
                raise NonTerminatingException("Command not executed. Switching to human mode")
            case _:
                raise NonTerminatingException(
                    f"Command not executed. The user rejected your command with the following message: {user_input}"
                )

    def _prompt_and_handle_special(self, prompt: str) -> str:
        """Prompts the user, takes care of /h (followed by requery) and sets the mode. Returns the user input."""
        console.print(prompt, end="")
        user_input = prompt_session.prompt("")
        if user_input == "/h":
            console.print(
                f"Current mode: [bold green]{self.config.mode}[/bold green]\n"
                f"[bold green]/y[/bold green] to switch to [bold yellow]yolo[/bold yellow] mode (execute LM commands without confirmation)\n"
                f"[bold green]/c[/bold green] to switch to [bold yellow]confirmation[/bold yellow] mode (ask for confirmation before executing LM commands)\n"
                f"[bold green]/u[/bold green] to switch to [bold yellow]human[/bold yellow] mode (execute commands issued by the user)\n"
            )
            return self._prompt_and_handle_special(prompt)
        if user_input in self._MODE_COMMANDS_MAPPING:
            if self.config.mode == self._MODE_COMMANDS_MAPPING[user_input]:
                return self._prompt_and_handle_special(
                    f"[bold red]Already in {self.config.mode} mode.[/bold red]\n{prompt}"
                )
            self.config.mode = self._MODE_COMMANDS_MAPPING[user_input]
            console.print(f"Switched to [bold green]{self.config.mode}[/bold green] mode.")
            return user_input
        return user_input

    def has_finished(self, output: dict[str, str]):
        try:
            return super().has_finished(output)
        except Submitted as e:
            if self.config.confirm_exit:
                console.print(
                    "[bold green]Agent wants to finish.[/bold green] "
                    "[green]Type a comment to give it a new task or press enter to quit.\n"
                    "[bold yellow]>[/bold yellow] ",
                    end="",
                )
                if new_task := self._prompt_and_handle_special("").strip():
                    raise NonTerminatingException(f"The user added a new task: {new_task}")
            raise e
