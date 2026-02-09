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
[Bubblewrap](https://github.com/containers/bubblewrap) is a low-level, unprivileged sandboxing tool for Linux that enables running applications
in isolated environments with restricted access to the operating system and user data.
This environment uses bubblewrap to execute commands in a sandboxed environment.

!!! warning
    This environment is experimental.

!!! warning
    This environment is not supported on Windows.
"""

import logging
import os
import platform
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BubblewrapEnvironmentConfig:
    cwd: str = ""
    """Working directory for the sandbox."""
    env: dict[str, str] = field(default_factory=dict)
    """Dictionary of environment variables to set in the sandbox."""
    timeout: int = 30
    """Timeout for the command in seconds."""
    executable: str = os.getenv("MSWEA_BUBBLEWRAP_EXECUTABLE", "bwrap")
    """Path to the bubblewrap executable."""
    wrapper_args: list[str] = field(
        default_factory=lambda: [
            "--unshare-user-try",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/bin",
            "/bin",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--ro-bind",
            "/etc",
            "/etc",
            "--tmpfs",
            "/tmp",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--new-session",
            "--setenv",
            "PATH",
            "/usr/local/bin:/usr/sbin:/usr/bin:/bin",
        ]
    )
    """Arguments to pass to the bubblewrap executable."""


class BubblewrapEnvironment:
    def __init__(
        self, *, config_class = BubblewrapEnvironmentConfig, logger = None, **kwargs
    ):
        """This class executes bash commands in a bubblewrap environment and a separate working
        directory for each environment. See `BubblewrapEnvironmentConfig` for kwargs.
        """
        self.logger = logger or logging.getLogger("minisweagent.environment")
        self.config = config_class(**kwargs)
        self.working_dir = Path(tempfile.gettempdir()) / f"minisweagent-{uuid.uuid4().hex[:8]}"
        self.working_dir.mkdir(parents=True)

    def execute(self, command, cwd = ""):
        """Execute a command in the bubblewrap environment and return the result as a dict."""
        cwd = cwd or self.config.cwd or str(self.working_dir)

        cmd = [self.config.executable] + self.config.wrapper_args + ["--bind", cwd, cwd, "--chdir", cwd]

        # Add environment variables
        for key, value in self.config.env.items():
            cmd.extend(["--setenv", key, value])

        cmd.extend(["bash", "-c", command])

        result = subprocess.run(
            cmd,
            text=True,
            timeout=self.config.timeout,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return {"output": result.stdout, "returncode": result.returncode}

    def cleanup(self):
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)

    def __del__(self):
        """Cleanup working_dir when object is destroyed."""
        self.cleanup()

    def get_template_vars(self):
        return asdict(self.config) | platform.uname()._asdict()
