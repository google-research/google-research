package com.googleresearch.bustle;

import static java.lang.Math.max;
import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.toCollection;

import com.google.common.collect.ImmutableList;
import com.google.gson.Gson;
import com.googleresearch.bustle.serialization.SerializationUtils;
import com.googleresearch.bustle.value.ConstantValue;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.OperationValue;
import com.googleresearch.bustle.value.Value;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Option;

/** Generates training data by running the enumerative search on random inputs. */
@Command(name = "GenerateData", mixinStandardHelpOptions = true)
final class GenerateData implements Runnable {

  @SuppressWarnings("FieldCanBeFinal") // picocli modifies value when flag is set
  @Option(
      names = {"--random_seed"},
      description = "Random seed to use for synthetic data generation.")
  private static int randomSeed = 0;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--num_searches"},
      description = "Number of enumerative searches to run.")
  private static int numSearches = 10;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--num_values_per_search"},
      description = "Number of values per search to extract.")
  private static int numValuesPerSearch = 100;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--training_data_file"},
      description = "JSON file containing the generated training data.")
  private static String trainingDataFile = "/tmp/training_data.json";

  @Mixin
  private static Synthesizer synthesizerMixin; // Defines lots of options.

  private static final String[] CHARSETS =
      new String[] {
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ", // Uppercase letters
        "abcdefghijklmnopqrstuvwxyz", // Lowercase letters
        "0123456789", // Digits
        " ", // Space
        ".,-+_@$/", // Common punctuation
      };

  public static String randomInput(Random randomGen) {
    int length = 1 + randomGen.nextInt(10);
    List<String> usableCharsets = new ArrayList<>();
    for (String charset : CHARSETS) {
      if (randomGen.nextDouble() < 0.25) {
        usableCharsets.add(charset);
      }
    }
    if (usableCharsets.isEmpty()) {
      usableCharsets.add(CHARSETS[1]); // Lowercase letters
    }

    char[] randomChars = new char[length];
    for (int i = 0; i < length; i++) {
      String charset = usableCharsets.get(randomGen.nextInt(usableCharsets.size()));
      randomChars[i] = charset.charAt(randomGen.nextInt(charset.length()));
    }
    return new String(randomChars);
  }

  // Produces random int in the range [lower, upper).
  public static int randomInt(Random randomGen, int lower, int upper) {
    return lower + randomGen.nextInt(upper - lower);
  }

  public static List<Example> getRandomExamples(Random randomGen) {
    int numExamples = randomInt(randomGen, 2, 5);
    int numInputs = randomInt(randomGen, 2, 7) / 2; // 40% 1 input, 40% 2 inputs, 20% 3 inputs.
    int numFormats = randomInt(randomGen, 1, 3);

    // formats[f][i] is a list of symbols for the i-th input in the f-th format.
    List<List<List<Integer>>> formats = new ArrayList<>();
    int maxSymbol = 0;
    for (int f = 0; f < numFormats; f++) {
      List<List<Integer>> format = new ArrayList<>();
      // Choose number of slots for each input.
      int numSymbols = 0;
      for (int i = 0; i < numInputs; i++) {
        int numSlots = randomInt(randomGen, 1, 4);
        format.add(ImmutableList.of(numSlots)); // Just put the number of slots here, temporarily.
        numSymbols += numSlots;
      }
      maxSymbol = max(maxSymbol, numSymbols - 1);
      // Choose symbols for each input.
      for (int i = 0; i < numInputs; i++) {
        int numSlots = format.get(i).get(0);
        format.set(i, new ArrayList<>());
        for (int slot = 0; slot < numSlots; slot++) {
          format.get(i).add(randomInt(randomGen, 0, numSymbols));
        }
      }
      formats.add(format);
    }
    // Choose some symbols to be "example-persistent".
    Map<Integer, String> examplePersistentSymbols = new HashMap<>();
    for (int s = 0; s <= maxSymbol; s++) {
      if (randomGen.nextDouble() < 0.25) {
        examplePersistentSymbols.put(s, randomInput(randomGen));
      }
    }
    // Create examples.
    ImmutableList<String> outputs = ImmutableList.of("~", "&", "=", "^");
    List<Example> examples = new ArrayList<>();
    for (int e = 0; e < numExamples; e++) {
      List<List<Integer>> format = formats.get(randomGen.nextInt(numFormats));
      Map<Integer, String> symbolMap = new HashMap<>(examplePersistentSymbols);
      List<String> inputs = new ArrayList<>();
      for (int i = 0; i < numInputs; i++) {
        String input = "";
        for (Integer symbol : format.get(i)) {
          symbolMap.computeIfAbsent(symbol, (Integer k) -> randomInput(randomGen));
          input += symbolMap.get(symbol);
        }
        inputs.add(input);
      }
      examples.add(Example.create(inputs, outputs.get(e)));
    }
    return examples;
  }

  public static List<List<DataItem>> generateData(
      int numSearches, int numValuesPerSearch, Random randomGen) {
    List<List<DataItem>> dataFromAllSearches = new ArrayList<>();

    for (int i = 0; i < numSearches; i++) {
      System.out.println("Performing search " + (i + 1) + " of " + numSearches);
      List<DataItem> dataFromThisSearch = new ArrayList<>();

      List<Example> examples = getRandomExamples(randomGen);
      Benchmark benchmark =
          Benchmark.createBenchmarkWithNoTags(
              "RandomlyGenerated",
              "Randomly generated for dataset creation",
              examples,
              examples,
              "Junk Expected Program");
      Set<Value> seenValues = new HashSet<>();
      Synthesizer.synthesize(benchmark, seenValues);

      // Keep only values that are Strings and the result of an operation.
      List<OperationValue> stringValues =
          seenValues.stream()
              .filter(v -> v.getType().equals(String.class) && v instanceof OperationValue)
              .map(v -> (OperationValue) v)
              .collect(toCollection(ArrayList::new));
      // remove stochasticity inherent in reading from Set
      stringValues.sort(comparing(Value::expression));
      Collections.shuffle(stringValues, randomGen);

      for (int j = 0; j < numValuesPerSearch; j++) {
        OperationValue selectedValue = stringValues.get(j);
        List<Object> outputs = selectedValue.getWrappedValues();
        List<Example> examplesWithOutputs = new ArrayList<>();
        for (int k = 0; k < examples.size(); k++) {
          examplesWithOutputs.add(
              Example.create(examples.get(k).inputs(), (String) outputs.get(k)));
        }
        dataFromThisSearch.add(DataItem.create(examplesWithOutputs, selectedValue));
      }

      dataFromAllSearches.add(dataFromThisSearch);
    }

    return dataFromAllSearches;
  }

  public static List<TrainingDataItem> buildTrainingDataItems(
      List<List<DataItem>> dataFromAllSearches, Random randomGen) {
    List<TrainingDataItem> trainingDataItems = new ArrayList<>();

    for (List<DataItem> dataFromOneSearch : dataFromAllSearches) {
      // We use this to check membership for negative examples
      List<Set<Value>> subExpressionSets = new ArrayList<>();
      // We use this so we can get the j-th subExpression at random.
      List<List<Value>> subExpressionLists = new ArrayList<>();
      // Contains data points not filtered out due to insufficient subExpressions
      List<DataItem> filteredData = new ArrayList<>();

      for (DataItem d : dataFromOneSearch) {
        Value v = d.getValue();
        Set<Value> subExpressions = Utils.getSubExpressions(v);

        // Remove InputValues, since these must always appear in the output expression (we assume).
        // Also remove ConstantValues because they would appear frequently in the training dataset
        // but appear extremely infrequently during evaluation.
        Set<Value> filteredSubExpressions =
            subExpressions.stream()
                .filter(s -> !(s instanceof InputValue) && !(s instanceof ConstantValue))
                .collect(toCollection(LinkedHashSet::new));

        // Remove the value itself, since the model won't ever see that as an input
        filteredSubExpressions.remove(v);

        // Only use this data point if there are subExpressions left post-filtering
        if (!filteredSubExpressions.isEmpty()) {
          filteredData.add(d);
          subExpressionSets.add(filteredSubExpressions);
          List<Value> subExpressionList = new ArrayList<>(filteredSubExpressions);
          // remove stochasticity inherent in reading from Set
          subExpressionList.sort(comparing(Value::expression));
          subExpressionLists.add(subExpressionList);
        }
      }

      // Each DataItem serves as the targetExpression for one positive and one negative
      // TrainingDataItem.
      for (int i = 0; i < filteredData.size(); i++) {
        // Choose a subexpression of the value.
        List<Value> subExpressions = subExpressionLists.get(i);
        Value subExpression = subExpressions.get(randomGen.nextInt(subExpressions.size()));
        Value targetExpression = filteredData.get(i).getValue();
        List<InputValue> inputValues =
            Utils.inputValuesFromExamplesList(filteredData.get(i).getExamples());

        // save out (input, output, subExpression, true)
        TrainingDataItem trueItem =
            new TrainingDataItem(inputValues, subExpression, targetExpression, true);
        trainingDataItems.add(trueItem);

        Value negativeSubExpression;
        while (true) {
          // choose a DataItem at random
          int dataIndex = randomGen.nextInt(subExpressionLists.size());
          // choose a sub-expression of that value
          int subExpressionIndex = randomGen.nextInt(subExpressionLists.get(dataIndex).size());

          negativeSubExpression = subExpressionLists.get(dataIndex).get(subExpressionIndex);
          if (!subExpressionSets.get(i).contains(negativeSubExpression)) {
            break;
          }
        }
        // save out (input, output, negativeSubExpression, false)
        TrainingDataItem falseItem =
            new TrainingDataItem(inputValues, negativeSubExpression, targetExpression, false);
        trainingDataItems.add(falseItem);
      }
    }
    return trainingDataItems;
  }

  @Override
  public void run() {
    Random randomGen = new Random();
    randomGen.setSeed(randomSeed); // Set the seed for reproducible generation
    List<List<DataItem>> data = generateData(numSearches, numValuesPerSearch, randomGen);
    List<TrainingDataItem> trainingDataItems = buildTrainingDataItems(data, randomGen);

    Gson gson = SerializationUtils.constructCustomGsonBuilder();
    try (PrintWriter writer = new PrintWriter(trainingDataFile, "UTF-8")) {
      for (TrainingDataItem tde : trainingDataItems) {
        String json = gson.toJson(tde, TrainingDataItem.class);
        writer.println(json);
      }
    } catch (IOException e) {
      System.out.println(
          "OH NO! We couldn't write the training data file " + trainingDataFile + ".");
    }
  }

  public static void main(String[] args) throws IOException {
    int exitCode = new CommandLine(new GenerateData()).execute(args);
    System.exit(exitCode);
  }

  private GenerateData() {}
}
