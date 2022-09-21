package com.googleresearch.bustle;

import com.google.common.collect.ImmutableList;
import java.util.Arrays;
import java.util.List;

/** A class to define a benchmark task. */
public class Benchmark {

  // The name of the benchmark.
  private final String name;
  // A brief text description of the benchmark.
  private final String description;
  // The input-output examples provided by the user to learn the transformation.
  private final ImmutableList<Example> trainExamples;
  // The examples used to test the synthesized program.
  private final ImmutableList<Example> testExamples;
  // The expected learnt program.
  private final String expectedProgram;
  // Tags to categorize benchmarks.
  private final ImmutableList<BenchmarkTag> tags;

  /** Construct a Benchmark with no tags. */
  public static Benchmark createBenchmarkWithNoTags(
      String name,
      String description,
      List<Example> trainExamples,
      List<Example> testExamples,
      String expectedProgram) {
    return new Benchmark(
        name,
        description,
        trainExamples,
        testExamples,
        expectedProgram,
        ImmutableList.of());
  }


  /** Constructs a Benchmark. */
  public Benchmark(
      String name,
      String description,
      List<Example> trainExamples,
      List<Example> testExamples,
      String expectedProgram,
      List<BenchmarkTag> tags) {
    this.name = name;
    this.description = description;
    this.trainExamples = ImmutableList.copyOf(trainExamples);
    this.testExamples = ImmutableList.copyOf(testExamples);
    this.expectedProgram = expectedProgram;
    this.tags = ImmutableList.copyOf(tags);
  }

  @Override
  public String toString() {
    return String.join(
        "\n",
        Arrays.asList(
            "Benchmark Name: " + name,
            "Description: " + description,
            "Train Examples: " + trainExamples,
            "Test Examples: " + testExamples,
            "Expected Program: " + expectedProgram));
  }

  public String getName() {
    return name;
  }

  public String getDescription() {
    return description;
  }

  public List<Example> getTrainExamples() {
    return trainExamples;
  }

  public List<Example> getTestExamples() {
    return testExamples;
  }

  public String getExpectedProgram() {
    return expectedProgram;
  }

  public List<BenchmarkTag> getTags() {
    return tags;
  }
}
