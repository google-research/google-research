#!/bin/bash
# Copyright 2025 The Google Research Authors.
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


# This script downloads necessary JAR files and compiles the code.

set -e

dir=$(dirname "$(readlink -f "$0")")
cd "${dir}"

echo "Downloading JAR files..."
mkdir -p lib
cd lib
wget https://repo1.maven.org/maven2/com/google/auto/value/auto-value/1.9/auto-value-1.9.jar
wget https://repo1.maven.org/maven2/com/google/auto/value/auto-value-annotations/1.9/auto-value-annotations-1.9.jar
wget https://repo1.maven.org/maven2/com/google/code/gson/gson/2.8.0/gson-2.8.0.jar
wget https://repo1.maven.org/maven2/com/google/guava/guava/31.1-jre/guava-31.1-jre.jar
wget https://repo1.maven.org/maven2/com/google/protobuf/protobuf-java/3.9.1/protobuf-java-3.9.1.jar
wget https://repo1.maven.org/maven2/com/google/truth/truth/1.1.3/truth-1.1.3.jar
wget https://repo1.maven.org/maven2/info/picocli/picocli/4.6.3/picocli-4.6.3.jar
wget https://repo1.maven.org/maven2/junit/junit/4.13.2/junit-4.13.2.jar
wget https://repo1.maven.org/maven2/org/apache/commons/commons-text/1.9/commons-text-1.9.jar
wget https://repo1.maven.org/maven2/org/hamcrest/hamcrest/2.2/hamcrest-2.2.jar
wget https://repo1.maven.org/maven2/org/tensorflow/libtensorflow/1.15.0/libtensorflow-1.15.0.jar
wget https://repo1.maven.org/maven2/org/tensorflow/libtensorflow_jni/1.15.0/libtensorflow_jni-1.15.0.jar
wget https://repo1.maven.org/maven2/org/tensorflow/proto/1.15.0/proto-1.15.0.jar
cd ..

echo "Compiling java files..."
javac -cp ".:lib/*" $(find . -name '*.java')

echo "Compilation finished!"
