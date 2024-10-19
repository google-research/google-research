# Copyright 2024 The Google Research Authors.
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

echo  "Compiling Misc.java"
javac -Xlint:unchecked Misc.java

echo  "Compiling BoolArray.java"
javac -Xlint:unchecked BoolArray.java

echo  "Compiling Utils.java"
javac -Xlint:unchecked Utils.java

echo  "Compiling History.java"
javac -Xlint:unchecked History.java

echo  "Compiling MemoryMonitor.java"
javac -Xlint:unchecked MemoryMonitor.java

echo  "Compiling LocalEmpiricalMeasure.java"
javac -Xlint:unchecked LocalEmpiricalMeasure.java

echo  "Compiling Feature.java"
javac -Xlint:unchecked Feature.java

echo  "Compiling Domain.java"
javac -Xlint:unchecked Domain.java

echo  "Compiling Observation.java"
javac -Xlint:unchecked Observation.java

echo  "Compiling Statistics.java"
javac -Xlint:unchecked Statistics.java

echo  "Compiling Dataset.java"
javac -Xlint:unchecked Dataset.java

echo  "Compiling Node.java"
javac -Xlint:unchecked Node.java

echo  "Compiling Tree.java"
javac -Xlint:unchecked Tree.java

echo  "Compiling MeasuredSupportAtTupleOfNodes.java"
javac -Xlint:unchecked MeasuredSupportAtTupleOfNodes.java

echo  "Compiling WeightedSupportAtTupleOfNodes.java"
javac -Xlint:unchecked WeightedSupportAtTupleOfNodes.java

echo  "Compiling GenerativeModelBasedOnEnsembleOfTrees.java"
javac -Xlint:unchecked GenerativeModelBasedOnEnsembleOfTrees.java

echo  "Compiling Algorithm.java"
javac -Xlint:unchecked Algorithm.java

echo  "Compiling Plotting.java"
javac -Xlint:unchecked Plotting.java

echo  "Compiling Wrapper.java"
javac -Xlint:unchecked Wrapper.java


