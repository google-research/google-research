#!/bin/bash
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


#
# Sets up a Python virtual environment, installs dependencies, and then runs
# either the unit tests or the Brainstorming Partner server.
#
# Usage:
#   ./run.sh                                          # Run the unit tests.
#   ./run.sh serve --project=YOUR_PROJECT_ID [flags]  # Start the server.
#
# With no arguments, runs server_test.py. The tests mock Speech-to-Text and
# Gemini, so they need no Cloud project or credentials. `serve` starts the
# server and passes the remaining arguments to server.py: --project is required
# and must name a Google Cloud project with the Speech-to-Text and Vertex AI
# APIs enabled. Run `./run.sh serve --helpshort` to list every flag.
#
# Prerequisites:
#   * Python 3.10 or newer, available as `python3`.
#   * To start the server, Application Default Credentials, created with:
#       gcloud auth application-default login
#
# Optional environment variables:
#   VENV_DIR                  Where to create the virtual environment (default:
#                             .venv next to this script).
#   SSL_CERT_FILE             CA bundle used to verify HTTPS connections
#                             (default: the certifi bundle installed in the
#                             virtual environment).
#   GRPC_ENABLE_FORK_SUPPORT  Whether gRPC registers pthread_atfork handlers
#                             (default: 0, disabled to avoid deadlocking when
#                             google-auth spawns `gcloud` subprocesses).

# Stop at the first failing command.
set -e

# Choose what to run before installing anything, so a usage error fails fast:
# no arguments runs the unit tests, and `serve [flags]` starts the server.
MODE="test"
if [ "$#" -gt 0 ]; then
  if [ "$1" = "serve" ]; then
    MODE="serve"
    shift
  else
    echo "Usage: $0                    Run the unit tests." >&2
    echo "       $0 serve --project=YOUR_PROJECT_ID [server.py flags]" >&2
    echo "                             Start the server." >&2
    exit 2
  fi
fi

# Print each command before running it.
set -x

# Run from this script's directory so relative paths work from anywhere.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Create the virtual environment on the first run so dependencies stay out of
# the system Python. Prefer `virtualenv` if it is installed; otherwise use the
# built-in `venv` module. If neither works (e.g. Debian without python3-venv),
# delete the partial directory and fall back to the current Python.
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/.venv}"
if [ ! -d "${VENV_DIR}/bin" ]; then
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv -p python3 "${VENV_DIR}"
  elif python3 -m venv "${VENV_DIR}" 2>/dev/null; then
    :
  else
    rm -rf "${VENV_DIR}"
  fi
fi

# Activate the virtual environment, if it exists, so `python3` and `pip` below
# use it.
if [ -f "${VENV_DIR}/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
fi

# Install the dependencies listed in requirements.txt. Re-runs are fast because
# pip skips requirements that are already satisfied.
if command -v pip >/dev/null 2>&1; then
  pip install -r "${SCRIPT_DIR}/requirements.txt"
elif python3 -m pip --version >/dev/null 2>&1; then
  python3 -m pip install -r "${SCRIPT_DIR}/requirements.txt"
fi

# Use certifi's CA bundle for HTTPS unless SSL_CERT_FILE is already set. Some
# Python installs (e.g. python.org's macOS installer) ship without CA
# certificates, so Gemini requests would fail with CERTIFICATE_VERIFY_FAILED.
if [ -z "${SSL_CERT_FILE:-}" ]; then
  CERT_PATH="$(python3 -c 'import certifi; print(certifi.where())' 2>/dev/null || true)"
  if [ -n "${CERT_PATH}" ]; then
    export SSL_CERT_FILE="${CERT_PATH}"
  fi
fi

# Disable gRPC's pthread_atfork handlers unless explicitly configured. When user
# Application Default Credentials have no embedded project_id, google-auth forks
# `gcloud` to look one up; if a gRPC polling cycle is active at the same time
# (especially on macOS), gRPC's fork handler blocks until the RPC deadline and
# freezes the server's event loop, dropping WebSocket heartbeats.
export GRPC_ENABLE_FORK_SUPPORT="${GRPC_ENABLE_FORK_SUPPORT:-0}"
export GRPC_VERBOSITY="${GRPC_VERBOSITY:-ERROR}"

# Start the server with the remaining flags, or run the offline unit tests.
if [ "${MODE}" = "serve" ]; then
  exec python3 "${SCRIPT_DIR}/server.py" "$@"
fi
exec python3 "${SCRIPT_DIR}/server_test.py"
