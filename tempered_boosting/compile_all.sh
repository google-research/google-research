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
javac Misc.java

echo  "Compiling QuickSort.java"
javac QuickSort.java

echo  "Compiling History.java"
javac History.java

echo  "Compiling MemoryMonitor.java"
javac MemoryMonitor.java

echo  "Compiling TemperedBoostException.java"
javac TemperedBoostException.java

echo  "Compiling Domain.java"
javac Domain.java

echo  "Compiling Example.java"
javac Example.java

echo  "Compiling Feature.java"
javac Feature.java

echo  "Compiling Statistics.java"
javac Statistics.java

echo  "Compiling Dataset.java"
javac Dataset.java

echo  "Compiling DecisionTreeNode.java"
javac DecisionTreeNode.java

echo  "Compiling DecisionTree.java"
javac DecisionTree.java

echo  "Compiling Algorithm.java"
javac Algorithm.java

echo  "Compiling Boost.java"
javac Boost.java

echo  "Compiling Utils.java"
javac Utils.java

echo  "Compiling PoincareDiskEmbeddingUI.java"
javac PoincareDiskEmbeddingUI.java

echo  "Compiling PoincareDiskEmbedding.java"
javac PoincareDiskEmbedding.java

echo  "Compiling JDecisionTreePane.java"
javac JDecisionTreePane.java

echo  "Compiling JDecisionTreeViewer.java"
javac -Xlint:deprecation JDecisionTreeViewer.java

echo  "Compiling Experiments.java"
javac Experiments.java

echo  "Compiling MonotonicTreeGraph.java"
javac MonotonicTreeGraph.java
