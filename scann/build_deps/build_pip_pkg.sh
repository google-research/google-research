# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

BAZEL_PREFIX="bazel-bin/build_pip_pkg.runfiles/_main/"
TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)
# set variable equal to "python" by default
: ${PYTHON:=python}

echo "$TMPDIR"

cp ${BAZEL_PREFIX}README.md "${TMPDIR}"
cp ${BAZEL_PREFIX}pyproject.toml "${TMPDIR}"
rsync -avm -L --exclude='*_test.py' ${BAZEL_PREFIX}scann "${TMPDIR}"

echo "from scann.scann_ops.py.scann_builder import ReorderType" >> "${TMPDIR}"/scann/__init__.py
echo "from scann.scann_ops.py.scann_builder import ScannBuilder" >> "${TMPDIR}"/scann/__init__.py
echo "from scann.scann_ops.py import scann_ops_pybind" >> "${TMPDIR}"/scann/__init__.py
# Attempt to import TensorFlow, which is required for ScaNN TF ops, but
# silently abort if this fails. TensorFlow and the ScaNN TF ops are optional
# components.
echo "try:" >> "${TMPDIR}"/scann/__init__.py
echo "  import tensorflow as _tf" >> "${TMPDIR}"/scann/__init__.py
echo "  from scann.scann_ops.py import scann_ops" >> "${TMPDIR}"/scann/__init__.py
echo "except ModuleNotFoundError:" >> "${TMPDIR}"/scann/__init__.py
echo "  pass" >> "${TMPDIR}"/scann/__init__.py
touch "${TMPDIR}"/scann/scann_ops/__init__.py
touch "${TMPDIR}"/scann/scann_ops/py/__init__.py
touch "${TMPDIR}"/scann/data_format/__init__.py
touch "${TMPDIR}"/scann/partitioning/__init__.py
touch "${TMPDIR}"/scann/proto/__init__.py
touch "${TMPDIR}"/scann/trees/__init__.py
touch "${TMPDIR}"/scann/trees/kmeans_tree/__init__.py

cwd="$(pwd)"
cd "$TMPDIR"
"$PYTHON" -m build --wheel "$@"
cd "$cwd"

cp "${TMPDIR}"/dist/*.whl ./
rm -rf "$TMPDIR"
