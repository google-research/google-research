set -e

echo "Compile Generative Decision Trees project"
javac -d compiled -classpath src src/*.java

echo "Prints the help"
java -classpath compiled Wrapper --help

echo "Download a copy of the Iris dataset"
wget https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset/iris.csv -O iris.csv

echo "Train and sample a new generator"
mkdir -p working_dir
java -classpath compiled Wrapper \
  --dataset=iris.csv \
  --work_dir=working_dir \
  --num_samples=1000 \
  --output_samples=working_dir/generated.csv \
  --output_stats=working_dir/statistics.stats

echo "Display some of the generated samples"
head working_dir/generated.csv
