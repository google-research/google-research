package com.googleresearch.bustle;

import com.google.common.collect.ImmutableList;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.util.List;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/** Prints benchmarks in JSON format. */
@Command(name = "BenchmarksAsJson", mixinStandardHelpOptions = true)
public class BenchmarksAsJson {

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
      names = {"--convert_sygus_benchmarks"},
      description = "Whether to use (only) SyGuS benchmarks.")
  private static boolean sygusBenchmarks = false;

  public static void main(String[] args) throws IOException {
    List<Benchmark> benchmarks;
    if (sygusBenchmarks) {
      benchmarks = SygusBenchmarks.getSygusBenchmarks();
    } else {
      benchmarks = Benchmarks.getBenchmarkWithName(benchmarkName, includedTags, excludedTags);
    }
    Gson gson = new GsonBuilder().disableHtmlEscaping().setPrettyPrinting().create();
    System.out.println(gson.toJson(benchmarks));
  }

  private BenchmarksAsJson() {}
}
