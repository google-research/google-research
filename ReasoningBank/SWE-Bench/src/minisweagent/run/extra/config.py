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

"""Utility to manage the global config file.

You can also directly edit the `.env` file in the config directory.

It is located at [bold green]{global_config_file}[/bold green].
"""

import os
import subprocess

from dotenv import set_key, unset_key
from prompt_toolkit import prompt
from rich.console import Console
from rich.rule import Rule
from typer import Argument, Typer

from minisweagent import global_config_file

app = Typer(
    help=__doc__.format(global_config_file=global_config_file),  # type: ignore
    no_args_is_help=True,
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console(highlight=False)


_SETUP_HELP = """To get started, we need to set up your global config file.

You can edit it manually or use the [bold green]mini-extra config set[/bold green] or [bold green]mini-extra config edit[/bold green] commands.

This setup will ask you for your model and an API key.

Here's a few popular models and the required API keys:

[bold green]anthropic/claude-sonnet-4-20250514[/bold green] ([bold green]ANTHROPIC_API_KEY[/bold green])
[bold green]openai/gpt-5[/bold green] or [bold green]openai/gpt-5-mini[/bold green] ([bold green]OPENAI_API_KEY[/bold green])
[bold green]gemini/gemini-2.5-pro[/bold green] ([bold green]GEMINI_API_KEY[/bold green])

[bold]Note: Please always include the provider in the model name.[/bold]

[bold yellow]You can leave any setting blank to skip it.[/bold yellow]

More information at https://mini-swe-agent.com/latest/quickstart/
To find the best model, check the leaderboard at https://swebench.com/
"""


def configure_if_first_time():
    if not os.getenv("MSWEA_CONFIGURED"):
        console.print(Rule())
        setup()
        console.print(Rule())


@app.command()
def setup():
    """Setup the global config file."""
    console.print(_SETUP_HELP.format(global_config_file=global_config_file))
    default_model = prompt(
        "Enter your default model (e.g., claude-sonnet-4-20250514): ", default=os.getenv("MSWEA_MODEL_NAME", "")
    ).strip()
    if default_model:
        set_key(global_config_file, "MSWEA_MODEL_NAME", default_model)
    console.print(
        "[bold yellow]If you already have your API keys set as environment variables, you can ignore the next question.[/bold yellow]"
    )
    key_name = prompt("Enter your API key name (e.g., ANTHROPIC_API_KEY): ").strip()
    key_value = None
    if key_name:
        key_value = prompt("Enter your API key value (e.g., sk-1234567890): ", default=os.getenv(key_name, "")).strip()
        if key_value:
            set_key(global_config_file, key_name, key_value)
    if not key_value:
        console.print(
            "[bold red]API key setup not completed.[/bold red] Totally fine if you have your keys as environment variables."
        )
    set_key(global_config_file, "MSWEA_CONFIGURED", "true")
    console.print(
        "\n[bold yellow]Config finished.[/bold yellow] If you want to revisit it, run [bold green]mini-extra config setup[/bold green]."
    )


@app.command()
def set(
    key: str | None = Argument(None, help="The key to set"),
    value: str | None = Argument(None, help="The value to set"),
):
    """Set a key in the global config file."""
    if key is None:
        key = prompt("Enter the key to set: ")
    if value is None:
        value = prompt(f"Enter the value for {key}: ")
    set_key(global_config_file, key, value)


@app.command()
def unset(key: str | None = Argument(None, help="The key to unset")):
    """Unset a key in the global config file."""
    if key is None:
        key = prompt("Enter the key to unset: ")
    unset_key(global_config_file, key)


@app.command()
def edit():
    """Edit the global config file."""
    editor = os.getenv("EDITOR", "nano")
    subprocess.run([editor, global_config_file])


if __name__ == "__main__":
    app()
