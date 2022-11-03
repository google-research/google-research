package com.googleresearch.bustle;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.lang.Math.max;
import static java.lang.Math.min;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.googleresearch.bustle.propertysignatures.ComputeSignature;
import com.googleresearch.bustle.propertysignatures.PropertySummary;
import com.googleresearch.bustle.value.ConstantValue;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.OutputValue;
import com.googleresearch.bustle.value.Value;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.stream.IntStream;
import org.apache.commons.text.similarity.LevenshteinDistance;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Option;

/** BUSTLE synthesizer. */
@Command(name = "Synthesizer", mixinStandardHelpOptions = true)
public final class Synthesizer implements Callable<List<SynthesisResult>> {

  @SuppressWarnings("FieldCanBeFinal") // picocli modifies value when flag is set
  @Option(
      names = {"--benchmark_name"},
      description = "Name of benchmark to run, or \"ALL\".")
  private static String benchmarkName = "ALL";

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--included_tags"},
      description = "Benchmark tags to include, by default everything.")
  private static ImmutableList<BenchmarkTag> includedTags = ImmutableList.of();

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--excluded_tags"},
      description = "Benchmark tags to exclude.")
  private static ImmutableList<BenchmarkTag> excludedTags =
      ImmutableList.of(
          BenchmarkTag.REGEX,
          BenchmarkTag.ARRAY,
          BenchmarkTag.TEXT_FORMATTING,
          BenchmarkTag.TOO_DIFFICULT,
          BenchmarkTag.SHOULD_FAIL);

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--sygus_benchmarks"},
      description = "Whether to use (only) SyGuS benchmarks.")
  private static boolean sygusBenchmarks = false;

  @Mixin
  private SygusBenchmarks sygusBenchmarksMixin; // Defines the --sygus_benchmarks_directory option.

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--output_file"},
      description = "Where to save the json file summarizing the results.")
  private static String outputFile = "/tmp/synthesis_results.json";

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--max_expressions"},
      description = "Maximum number of expressions to try.")
  private static int maxExpressions = 1000 * 1000 * 1000;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--time_limit"},
      description = "Maximum number of seconds to run.")
  private static double timeLimit = 10.0;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--max_weight"},
      description = "Maximum weight of expressions to try.")
  private static int maxWeight = 100;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--print_progress"},
      description = "Whether to print synthesis progress.")
  private static boolean printProgress = false;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--log_predictions"},
      description = "Whether to log predictions to a separate file.")
  private static boolean logPredictions = false;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--prediction_log_directory"},
      description = "Where to log model predictions.")
  private static String predictionLogDirectory = "/tmp/prediction_logs/";

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--model_reweighting"},
      description = "Whether to use a learned model to reweight intermediate values.")
  private static boolean modelReweighting = true;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--model_reweighting_fraction"},
      description =
          "Perform model reweighting only for the first fraction of the synthesis, either by time"
              + " or number of expressions.")
  private static double modelReweightingFraction = 1.0;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--model_directory"},
      description = "Directory for saved model.")
  private static String modelDirectory = "/tmp/saved_model_dir/";

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--model_batch_size"},
      description = "Batch size for the model.")
  private static int modelBatchSize = 1024;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--reweight_alpha"},
      description = "Alpha in reweighting scheme.")
  private static double reweightAlpha = 0.6;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--reweight_beta"},
      description = "Beta in reweighting scheme.")
  private static double reweightBeta = 0.4;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--premise_selection"},
      description = "Whether to use learned model to exclude some operations.")
  private static boolean premiseSelection = false;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--premise_selection_model_directory"},
      description = "Directory for saved model.")
  private static String premiseSelectionModelDirectory = "/tmp/saved_premise_selection_model_dir/";

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--premises_to_drop"},
      description = "Number of premises to discard.")
  private static int premisesToDrop = 2;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--heuristic_reweighting"},
      description =
          "Whether to use substring and edit distance heuristics to reweight intermediate values.")
  private static boolean heuristicReweighting = true;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--output_substring_weight_reduction"},
      description = "Weight subtracted from values that are substrings of the output.")
  private static int outputSubstringWeightReduction = 2;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--edit_distance_weight_addition"},
      description = "Maximum weight added to values that have large edit distance to the output.")
  private static int editDistanceWeightAddition = 3;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--parse_flags_only"},
      description = "Whether to only parse flags and skip synthesis (for testing).")
  private static boolean parseFlagsOnly = false;

  @SuppressWarnings("FieldCanBeFinal")
  @Option(
      names = {"--quick_test"},
      description = "Whether to only synthesize the first 3 tasks (for testing).")
  private static boolean quickTest = false;

  // Constants that are always used.
  private static final ImmutableList<Object> ALWAYS_USED_CONSTANTS =
      ImmutableList.of("", 0, 1, 2, 3, 99);
  // These string constants are used if they appear in an input or output string.
  private static final ImmutableList<String> CANDIDATE_STRING_CONSTANTS =
      ImmutableList.of(
          " ", ",", ".", "!", "?", "(", ")", "[", "]", "<", ">", "{", "}", "-", "+", "_", "/", "$",
          "#", ":", ";", "@", "%", "0");

  /** Fills the container arguments to prepare for a bottom-up enumerative synthesis search. */
  private static void synthesisSetup(
      Benchmark benchmark,
      List<List<? extends Object>> inputObjects,
      List<InputValue> inputValues,
      List<Object> outputObjects,
      List<List<Value>> valuesByWeight,
      Set<Value> seenValues) {
    int numExamples = benchmark.getTrainExamples().size();
    int numInputs = benchmark.getTrainExamples().get(0).inputs().size();

    for (Example ex : benchmark.getTrainExamples()) {
      if (ex.inputs().size() != numInputs) {
        throw new IllegalArgumentException(
            "Benchmark's examples must all have the same number of inputs.");
      }
      inputObjects.add(ex.inputs());
      outputObjects.add(ex.output());
    }

    for (int i = 0; i < numInputs; i++) {
      List<Object> thisInputObjects = new ArrayList<>();
      for (int j = 0; j < numExamples; j++) {
        thisInputObjects.add(inputObjects.get(j).get(i));
      }
      inputValues.add(new InputValue(thisInputObjects, "var_" + i));
    }
    OutputValue outputValue = new OutputValue(outputObjects);

    valuesByWeight.add(ImmutableList.of()); // Nothing has weight 0.

    ImmutableList<Value> alwaysUsedConstantValues =
        ALWAYS_USED_CONSTANTS.stream()
            .map(c -> new ConstantValue(c, numExamples))
            .collect(toImmutableList());
    ImmutableList<Value> extractedConstantValues =
        ConstantExtraction.extractConstants(inputValues, outputValue, CANDIDATE_STRING_CONSTANTS)
            .stream()
            .map(c -> new ConstantValue(c, numExamples))
            .collect(toImmutableList());
    System.out.println("Extracted constants: " + extractedConstantValues);

    List<Value> weightOneValues = new ArrayList<>();
    weightOneValues.addAll(inputValues);
    weightOneValues.addAll(alwaysUsedConstantValues);
    weightOneValues.addAll(extractedConstantValues);

    valuesByWeight.add(weightOneValues);
    seenValues.addAll(weightOneValues);

    for (int weight = 2; weight <= maxWeight; weight++) {
      valuesByWeight.add(new ArrayList<>());
    }
  }

  private static int heuristicReweightResult(
      Value result, int weight, int numExamples, Class<?> outputType, List<Object> outputObjects) {
    int resultWeight = weight;

    if (outputType.equals(String.class) && result.getType().equals(String.class)) {
      // Check for substrings, and increase weight of values that are very different from
      // the output.

      // If the weight reduction is zero, then don't bother checking for substrings at all.
      boolean allResultsInOutputs = outputSubstringWeightReduction > 0;

      double maxNormalizedDistance = 0.0;
      for (int i = 0; i < numExamples; i++) {
        String outputString = (String) outputObjects.get(i);
        String resultString = (String) result.getWrappedValue(i);

        if (allResultsInOutputs && !outputString.contains(resultString)) {
          allResultsInOutputs = false;
        }

        if (editDistanceWeightAddition > 0) {
          // Normalized distance: 2 * lev(x, y) / (len(x) + len(y) + lev(x, y)).
          int distance = LevenshteinDistance.getDefaultInstance().apply(resultString, outputString);
          double normalizedDistance =
              2.0 * distance / (outputString.length() + resultString.length() + distance);
          maxNormalizedDistance = max(maxNormalizedDistance, normalizedDistance);
        }
      }

      if (allResultsInOutputs) {
        resultWeight -= outputSubstringWeightReduction;
      }
      resultWeight += 1 + (int) (maxNormalizedDistance * editDistanceWeightAddition);
    }

    return max(resultWeight, weight);
  }

  /**
   * Checks if a value is too large to include in the search. Very long strings can lead to slowdown
   * when evaluating operations, and very large ints can as well for operations like REPT.
   */
  private static boolean valueTooLarge(Value value) {
    if (value.getType().equals(String.class)) {
      for (Object str : value.getWrappedValues()) {
        if (((String) str).length() >= 99) { // Make sure to throw out REPT(X, 99)
          return true;
        }
      }
    } else if (value.getType().equals(Integer.class)) {
      for (Object val : value.getWrappedValues()) {
        if (Math.abs((Integer) val) >= 100) {
          return true;
        }
      }
    }
    return false;
  }

  private static double sigmoid(float x) {
    if (x >= 0) {
      return 1 / (1 + Math.exp(-x));
    } else {
      double z = Math.exp(x);
      return z / (1 + z);
    }
  }

  private static List<Operation> selectPremises(
      SavedModelWrapper premiseSelectionModel, List<PropertySummary> exampleSignature) {
    List<Operation> allOperations = Operation.getOperations();
    List<Operation> operations = new ArrayList<>();
    if (premiseSelectionModel == null) {
      operations = allOperations;
    } else {
      List<PropertySummary> dummySignature =
          ComputeSignature.computeValueSignature(null, new OutputValue(ImmutableList.of("dummy")));
      List<PropertySummary> zeroSignature =
          Collections.nCopies(dummySignature.size(), PropertySummary.ALL_TRUE);
      float[][] premiseModelResults =
          premiseSelectionModel.doInference(exampleSignature, ImmutableList.of(zeroSignature));
      // indicesByScore is a list of indices - one-per-operation - sorted by model's estimated
      // likelihood of that operation appearing in the solution.
      // Note: the outputs of the premiseSelection model are in fact probabilties and not logits,
      // due to some keras grossness.
      ImmutableList<Integer> indicesByScore =
          IntStream.range(0, allOperations.size()).boxed().collect(toImmutableList());
      Collections.sort(indicesByScore, Comparator.comparing(idx -> premiseModelResults[0][idx]));
      // We can then just throw out the operations whose indices appear first, as they are the ones
      // with the lowest estimated likelihood.
      for (int i = 0; i < allOperations.size(); i++) {
        if (indicesByScore.indexOf(i) >= premisesToDrop) {
          operations.add(allOperations.get(i));
        }
      }
    }
    return operations;
  }

  /**
   * Bottom-up enumerative search. Expressions are enumerated in order of increasing weight. All
   * inputs, constants, and operations have weight 1.
   */
  public static SynthesisResult synthesize(Benchmark benchmark) {
    return synthesize(benchmark, new HashSet<>());
  }

  public static SynthesisResult synthesize(Benchmark benchmark, Set<Value> seenValues) {
    return synthesize(benchmark, null, null, seenValues);
  }

  public static SynthesisResult synthesize(
      Benchmark benchmark, SavedModelWrapper model, SavedModelWrapper premiseSelectionModel) {
    return synthesize(benchmark, model, premiseSelectionModel, new HashSet<>());
  }

  public static SynthesisResult synthesize(
      Benchmark benchmark,
      SavedModelWrapper model,
      SavedModelWrapper premiseSelectionModel,
      Set<Value> seenValues) {
    long totalCoreTime = 0;
    long totalSignatureTime = 0;
    long totalModelTime = 0;
    long totalHeuristicTime = 0;

    long startTime = System.nanoTime();
    int numExamples = benchmark.getTrainExamples().size();
    if (numExamples == 0) {
      throw new IllegalArgumentException("Benchmark must have at least one example.");
    }
    // Used for storing model predictions for further analysis (insertion is flag-guarded)
    List<PredictionDataItem> predictionDataItems = new ArrayList<>();

    List<List<? extends Object>> inputObjects = new ArrayList<>(); // numExamples x numInputs.
    List<InputValue> inputValues = new ArrayList<>();
    List<Object> outputObjects = new ArrayList<>();
    // Implicitly a map from weight to list of values of that weight, with weight encoded by index.
    List<List<Value>> valuesByWeight = new ArrayList<>();

    synthesisSetup(benchmark, inputObjects, inputValues, outputObjects, valuesByWeight, seenValues);

    OutputValue outputValue = new OutputValue(outputObjects);
    Class<?> outputType = outputValue.getType();

    List<PropertySummary> exampleSignature =
        ComputeSignature.computeExampleSignature(inputValues, outputValue);

    // Choose which operations to exclude based on the premise selector
    List<Operation> operations = selectPremises(premiseSelectionModel, exampleSignature);

    int numExpressionsTried = 0;

    weightLoop:
    for (int weight = 2; weight <= maxWeight; weight++) {
      if (printProgress) {
        System.out.println("Searching weight " + weight);
      }

      // Find new values with this weight.
      List<Value> newValues = new ArrayList<>();
      long coreStart = System.nanoTime();
      for (Operation operation : operations) {
        int numArgs = operation.getNumArgs();
        List<Class<?>> argTypes = operation.getArgTypes();

        for (List<Integer> argWeights : Utils.generatePartitions(weight - 1, numArgs)) {
          if ((System.nanoTime() - startTime) / 1e9 > timeLimit) {
            totalCoreTime += System.nanoTime() - coreStart;
            break weightLoop;
          }

          List<List<Value>> allArgChoices = new ArrayList<>();
          for (int argIndex = 0; argIndex < numArgs; argIndex++) {
            final int finalArgIndex = argIndex;
            allArgChoices.add(
                valuesByWeight.get(argWeights.get(argIndex)).stream()
                    .filter(v -> v.getType().equals(argTypes.get(finalArgIndex)))
                    .collect(toImmutableList()));
          }

          for (List<Value> argList : Lists.cartesianProduct(allArgChoices)) {
            if (numExpressionsTried >= maxExpressions) {
              totalCoreTime += System.nanoTime() - coreStart;
              break weightLoop;
            }

            Value result = operation.apply(argList);
            numExpressionsTried++;

            if (result == null || valueTooLarge(result)) {
              continue;
            }
            if (seenValues.add(result)) {
              newValues.add(result);
            }
            if (result.equals(outputValue)) {
              String solution = result.expression();
              System.out.println("Synthesis success! " + solution);
              System.out.println("Num expressions tried: " + numExpressionsTried);
              System.out.println("Num unique values: " + seenValues.size());
              long endTime = System.nanoTime();
              double elapsedTime = (endTime - startTime) / 1e9;
              if (logPredictions) {
                Utils.analyzePredictions(
                    true, result, predictionDataItems, predictionLogDirectory, benchmark);
              }
              return SynthesisResult.createSuccess(
                  benchmark.getName(),
                  numExpressionsTried,
                  seenValues.size(),
                  elapsedTime,
                  solution,
                  weight);
            }
          }
        }
      }
      totalCoreTime += System.nanoTime() - coreStart;

      if (printProgress) {
        System.out.println(
            "  Found " + newValues.size() + " new values while searching weight " + weight);
      }

      // Reweight the new values using the model.
      int[] modelWeightDeltas = new int[newValues.size()];
      double[] probabilities = new double[newValues.size()]; // strictly for analysis
      double currentSynthesisFraction =
          max(
              numExpressionsTried / (double) maxExpressions,
              (System.nanoTime() - startTime) / 1e9 / timeLimit);
      boolean skipModel = currentSynthesisFraction > modelReweightingFraction;
      if (skipModel && printProgress) {
        System.out.println("Skipping signatures and model runs for weight " + weight);
      }
      if (model == null || skipModel) {
        Arrays.fill(modelWeightDeltas, 0);
        Arrays.fill(probabilities, 0);
      } else {
        int numValuesProcessed = 0;
        while (numValuesProcessed < newValues.size()) {
          if ((System.nanoTime() - startTime) / 1e9 > timeLimit) {
            break weightLoop;
          }
          int thisBatchSize = min(modelBatchSize, newValues.size() - numValuesProcessed);

          long signatureStart = System.nanoTime();

          // Compute property signatures with a parallel stream
          ImmutableList<List<PropertySummary>> valueSignatures =
              IntStream.range(numValuesProcessed, numValuesProcessed + thisBatchSize)
                  .boxed()
                  .collect(toImmutableList())
                  .stream()
                  .parallel()
                  .map(i -> ComputeSignature.computeValueSignature(newValues.get(i), outputValue))
                  .collect(toImmutableList());

          totalSignatureTime += System.nanoTime() - signatureStart;
          long modelStart = System.nanoTime();
          float[][] modelResults = model.doInference(exampleSignature, valueSignatures);
          for (int i = 0; i < thisBatchSize; i++) {
            float logit = modelResults[i][0];
            int modelWeightDelta = 0;
            double probability = sigmoid(logit);

            // Optionally pre-process predictions for out-of-band analysis
            // Begin Analysis Code
            if (logPredictions) {
              predictionDataItems.add(
                  new PredictionDataItem(
                      benchmark.getExpectedProgram(),
                      newValues.get(numValuesProcessed + i),
                      newValues.get(numValuesProcessed + i).expression(),
                      true, // as a placeholder, set isPositiveExample to true
                      probability,
                      exampleSignature,
                      valueSignatures.get(i)));
            }
            // End Analysis Code

            if (probability > reweightAlpha) {
              modelWeightDelta = 0;
            } else if (probability > reweightBeta) {
              modelWeightDelta = 1;
            } else {
              // probability in [0, 0.4]
              // (0.4 - probability) in [0, 0.4]
              // (10 * (0.4 - probability) + 2) in [2, 6]
              // Due to integer truncation, delta=6 won't happen unless probability is exactly 0.
              modelWeightDelta = (int) (10 * (0.4 - probability) + 2);
            }
            modelWeightDeltas[numValuesProcessed + i] = modelWeightDelta;
            probabilities[numValuesProcessed + i] = probability;
          }
          totalModelTime += System.nanoTime() - modelStart;
          numValuesProcessed += thisBatchSize;
        }
      }

      // Reweight the new values using heuristics.
      int[] heuristicWeights = new int[newValues.size()];
      if (!heuristicReweighting) {
        Arrays.fill(heuristicWeights, weight);
      } else {
        long heuristicStart = System.nanoTime();
        for (int i = 0; i < newValues.size(); i++) {
          heuristicWeights[i] =
              heuristicReweightResult(
                  newValues.get(i), weight, numExamples, outputType, outputObjects);
        }
        totalHeuristicTime += System.nanoTime() - heuristicStart;
      }

      // Record the new values using the updated weights.
      for (int i = 0; i < newValues.size(); i++) {
        int newValueWeight = heuristicWeights[i] + modelWeightDeltas[i];
        if (newValueWeight <= maxWeight) {
          valuesByWeight.get(newValueWeight).add(newValues.get(i));
        }
      }
    }

    System.out.println("Synthesis failure.");
    System.out.println("Total core time: " + totalCoreTime / 1e9 + " sec");
    System.out.println("Total signature time: " + totalSignatureTime / 1e9 + " sec");
    System.out.println("Total model time: " + totalModelTime / 1e9 + " sec");
    System.out.println("Total heuristic time: " + totalHeuristicTime / 1e9 + " sec");

    return SynthesisResult.createFailure(
        benchmark.getName(),
        numExpressionsTried,
        seenValues.size(),
        (System.nanoTime() - startTime) / 1e9);
  }

  public static void warmUp(SavedModelWrapper model) {
    // Warm up by synthesizing a benchmark.
    System.out.println("Warming up...");
    synthesize(Benchmarks.prependMr(), model, null);
    System.out.println("\nDone warming up.\n");
  }

  private static double mean(List<? extends Number> numbers) {
    return numbers.stream().mapToDouble(Number::doubleValue).average().orElse(Double.NaN);
  }

  private static double geometricMean(List<? extends Number> numbers) {
    return Math.exp(
        numbers.stream().mapToDouble(x -> Math.log(x.doubleValue())).average().orElse(Double.NaN));
  }

  private static Map<String, Object> saveResultsToJson(List<SynthesisResult> synthesisResults)
      throws IOException {
    int numSolved = 0;
    int numBenchmarks = synthesisResults.size();
    List<Double> solveTimes = new ArrayList<>();
    List<Integer> solveNumExpressions = new ArrayList<>();
    List<Integer> solveNumUnique = new ArrayList<>();
    List<Double> allTimes = new ArrayList<>();
    List<Integer> allNumExpressions = new ArrayList<>();
    List<Integer> allNumUnique = new ArrayList<>();
    for (SynthesisResult result : synthesisResults) {
      if (result.getSuccess()) {
        numSolved++;
        solveTimes.add(result.getElapsedTime());
        solveNumExpressions.add(result.getNumExpressionsTried());
        solveNumUnique.add(result.getNumUniqueValues());
      }
      allTimes.add(result.getElapsedTime());
      allNumExpressions.add(result.getNumExpressionsTried());
      allNumUnique.add(result.getNumUniqueValues());
    }

    // Maintain insertion-order of entries for consistent JSON output.
    Map<String, Object> jsonMap = new LinkedHashMap<>();
    jsonMap.put("numSolved", numSolved);
    jsonMap.put("numBenchmarks", numBenchmarks);
    jsonMap.put("solveTimesMean", mean(solveTimes));
    jsonMap.put("solveTimesGeoMean", geometricMean(solveTimes));
    jsonMap.put("solveNumExpressionsMean", mean(solveNumExpressions));
    jsonMap.put("solveNumExpressionsGeoMean", geometricMean(solveNumExpressions));
    jsonMap.put("solveNumUniqueMean", mean(solveNumUnique));
    jsonMap.put("solveNumUniqueGeoMean", geometricMean(solveNumUnique));
    jsonMap.put("allTimesMean", mean(allTimes));
    jsonMap.put("allTimesGeoMean", geometricMean(allTimes));
    jsonMap.put("allNumExpressionsMean", mean(allNumExpressions));
    jsonMap.put("allNumExpressionsGeoMean", geometricMean(allNumExpressions));
    jsonMap.put("allNumUniqueMean", mean(allNumUnique));
    jsonMap.put("allNumUniqueGeoMean", geometricMean(allNumUnique));
    jsonMap.put("results", synthesisResults);

    Gson gson =
        new GsonBuilder()
            .disableHtmlEscaping()
            .setPrettyPrinting()
            .serializeSpecialFloatingPointValues()
            .create();
    try (PrintWriter writer = new PrintWriter(outputFile, "UTF-8")) {
      writer.println(gson.toJson(jsonMap));
      writer.close();
    } catch (IOException e) {
      System.out.println("OH NO! We couldn't write the results file.");
      throw e;
    }

    return jsonMap;
  }

  @Override
  public List<SynthesisResult> call() throws IOException {
    if (parseFlagsOnly) {
      return new ArrayList<>();
    }

    SavedModelWrapper model = modelReweighting ? new SavedModelWrapper(modelDirectory) : null;

    SavedModelWrapper premiseSelectionModel =
        premiseSelection ? new SavedModelWrapper(premiseSelectionModelDirectory) : null;

    warmUp(model);

    List<Benchmark> benchmarks;
    if (sygusBenchmarks) {
      benchmarks = SygusBenchmarks.getSygusBenchmarks();
    } else {
      benchmarks = Benchmarks.getBenchmarkWithName(benchmarkName, includedTags, excludedTags);
    }

    List<SynthesisResult> synthesisResults = new ArrayList<>();
    for (Benchmark b : benchmarks) {
      if (quickTest && synthesisResults.size() >= 3) {
        break;
      }
      System.gc();
      System.out.println("--------------------------------------------------------------");
      System.out.println("Attempting benchmark: " + b);
      long startTime = System.nanoTime();
      SynthesisResult synthesisResult = synthesize(b, model, premiseSelectionModel);
      long endTime = System.nanoTime();
      System.out.println("Elapsed time: " + (endTime - startTime) / 1e9 + " seconds");
      synthesisResults.add(synthesisResult);
    }

    System.out.println("Writing synthesis results to: " + outputFile);
    Map<String, Object> jsonMap = saveResultsToJson(synthesisResults);
    System.out.printf(
        "\nSolved %d / %d benchmarks.\n",
        (int) jsonMap.get("numSolved"),
        (int) jsonMap.get("numBenchmarks"));
    System.out.printf(
        "Arithmetic mean of successes: %5.2f seconds, %10.1f expressions, %9.1f unique values\n",
        (double) jsonMap.get("solveTimesMean"),
        (double) jsonMap.get("solveNumExpressionsMean"),
        (double) jsonMap.get("solveNumUniqueMean"));
    System.out.printf(
        "Geometric mean of successes:  %5.2f seconds, %10.1f expressions, %9.1f unique values\n",
        (double) jsonMap.get("solveTimesGeoMean"),
        (double) jsonMap.get("solveNumExpressionsGeoMean"),
        (double) jsonMap.get("solveNumUniqueGeoMean"));

    return synthesisResults;
  }

  public static void main(String[] args) {
    int exitCode = new CommandLine(new Synthesizer()).execute(args);
    System.exit(exitCode);
  }
}
