package com.googleresearch.bustle;

import com.google.gson.Gson;
import com.googleresearch.bustle.exception.SynthesisError;
import com.googleresearch.bustle.serialization.SerializationUtils;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.OperationValue;
import com.googleresearch.bustle.value.Value;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/** Utility functions. */
public final class Utils {

  private static void chooseDividers(
      int min, int max, int numDividers, List<Integer> selection, List<List<Integer>> results) {
    if (selection.size() == numDividers) {
      results.add(new ArrayList<>(selection));
      return;
    }
    for (int i = min; i <= max; i++) {
      selection.add(i);
      chooseDividers(i + 1, max, numDividers, selection, results);
      selection.remove(selection.size() - 1);
    }
  }

  /**
   * Computes partitions of total into numParts positive parts.
   *
   * <p>For example, total=5 can be partitioned into numParts=3 positive parts in the following
   * ways: [1, 1, 3], [1, 2, 2], [1, 3, 1], [2, 1, 2], [2, 2, 1], [3, 1, 1].
   */
  public static List<List<Integer>> generatePartitions(int total, int numParts) {
    if (total < 0 || numParts <= 0) {
      throw new SynthesisError(
          "In generatePartitions, total must be nonnegative, and " + "numParts must be positive.");
    }
    List<List<Integer>> partitions = new ArrayList<>();
    List<List<Integer>> dividerChoices = new ArrayList<>();
    chooseDividers(1, total - 1, numParts - 1, new ArrayList<Integer>(), dividerChoices);
    for (List<Integer> dividers : dividerChoices) {
      List<Integer> partition = new ArrayList<>(numParts);
      int lastDivider = 0;
      for (int divider : dividers) {
        partition.add(divider - lastDivider);
        lastDivider = divider;
      }
      partition.add(total - lastDivider);
      partitions.add(partition);
    }
    return partitions;
  }

  public static List<InputValue> inputValuesFromExamplesList(List<Example> examples) {

    // numExamples x numInputs.
    List<List<? extends Object>> inputObjects = new ArrayList<>();
    int numExamples = examples.size();
    int numInputs = examples.get(0).inputs().size();

    for (Example ex : examples) {
      if (ex.inputs().size() != numInputs) {
        throw new IllegalArgumentException(
            "Benchmark's examples must all have the same number of inputs.");
      }
      inputObjects.add(ex.inputs());
    }

    List<InputValue> inputValues = new ArrayList<>();
    for (int i = 0; i < numInputs; i++) {
      List<Object> thisInputObjects = new ArrayList<>();
      for (int j = 0; j < numExamples; j++) {
        thisInputObjects.add(inputObjects.get(j).get(i));
      }
      inputValues.add(new InputValue(thisInputObjects, "var_" + i));
    }

    return inputValues;
  }

  public static Set<Value> getSubExpressions(Value value) {
    // Return all possible sub-expressions of a Value. Recursive.
    // NOTE: the returned set includes the value itself.
    Set<Value> subExpressions = new HashSet<>();
    subExpressions.add(value);
    if (value instanceof OperationValue) {
      // If value is an operationValue, recurse into all arguments
      for (Value arg : ((OperationValue) value).getArguments()) {
        subExpressions.addAll(getSubExpressions(arg));
      }
    }
    return subExpressions;
  }

  public static void analyzePredictions(
      boolean benchmarkSolved,
      Value result,
      List<PredictionDataItem> predictionDataItems,
      String predictionsDir,
      Benchmark benchmark) {

    Random randomGen = new Random();
    Set<Value> solutionSubExpressions = getSubExpressions(result);
    Gson gson = SerializationUtils.constructCustomGsonBuilder();
    String extension = benchmarkSolved ? "_success" : "_failure";
    String filename = predictionsDir + benchmark.getName() + extension;
    int printCount = 0;
    try (PrintWriter writer = new PrintWriter(filename, "UTF-8")) {
      for (PredictionDataItem pdi : predictionDataItems) {
        boolean isPositiveExample =
            (solutionSubExpressions.contains(pdi.getSubExpression())
                && benchmarkSolved); // if benchmark wasn't solved, subExpressions not useful
        pdi.setIsPositiveExample(isPositiveExample);
        String jsonResult = gson.toJson(pdi);
        if (!isPositiveExample && (randomGen.nextDouble() > 0.05) && (printCount > 10)) {
          continue; // randomly skip 95% of negatives, after we've printed at least 10
        }
        writer.println(jsonResult);
        printCount++;
      }
      if (benchmarkSolved) {
        // add dummy PDI corresponding to solution
        // this is so our analyzer can see the solution the model actually came up with, so that it
        // can be compared to the expectedProgram
        PredictionDataItem dummyPDI = predictionDataItems.get(0);
        PredictionDataItem solutionPDI =
            new PredictionDataItem(
                benchmark.getExpectedProgram(),
                result, // this is the solution
                result.expression(),
                true,
                10000.0, // This is a sentinel value, signifying that this value is the solution
                dummyPDI.getExampleSignature(),
                dummyPDI.getValueSignature());
        String jsonResult = gson.toJson(solutionPDI);
        writer.println(jsonResult);
      }
    } catch (IOException e) {
      System.out.println(
          "OH NO! We couldn't write to the prediction data dir" + predictionsDir + ".");
    }
  }

  // Given a value, return a list of booleans with as many elements as there are operations.
  // The i-th element of this list will be true iff the i-th operation is present in the value.
  public static List<Boolean> getOperationPresenceListFromValue(Value value) {
    List<Boolean> operationPresenceList = new ArrayList<>();

    Set<Value> subExpressions = getSubExpressions(value);
    Set<Pair<String, Integer>> operationArgumentCountPairs = new HashSet<>();
    for (Value subExpression : subExpressions) {
      if (subExpression instanceof OperationValue) {
        OperationValue operationValue = (OperationValue) subExpression;
        Pair<String, Integer> thisPair =
            new Pair<>(
                operationValue.getOperation().getName(),
                operationValue.getOperation().getNumArgs());
        operationArgumentCountPairs.add(thisPair);
      }
    }
    for (Operation operation : Operation.getOperations()) {
      Pair<String, Integer> thisPair = new Pair<>(operation.getName(), operation.getNumArgs());
      operationPresenceList.add(operationArgumentCountPairs.contains(thisPair));
    }
    return operationPresenceList;
  }

  private Utils() {}
}
