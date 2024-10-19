package com.googleresearch.bustle;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.io.Files;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;
import java.util.Queue;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import picocli.CommandLine.Option;

/**
 * Creates Benchmark objects for SyGuS tasks.
 *
 * <p>To get the benchmarks:
 *
 * <pre>
 * cd ~/bustle/sygus_benchmarks
 * svn checkout https://github.com/SyGuS-Org/benchmarks/trunk/comp/2019/PBE_SLIA_Track/euphony
 * svn checkout https://github.com/ellisk42/ec/trunk/PBE_Strings_Track
 * </pre>
 */
final class SygusBenchmarks {

  @SuppressWarnings("FieldCanBeFinal") // picocli modifies value when flag is set
  @Option(
      names = {"--sygus_benchmarks_directory"},
      description = "Directory for SyGuS benchmarks.")
  private static String sygusBenchmarksDirectory = "~/bustle/sygus_benchmarks";

  private static final String SL_EXTENSION = ".sl";
  private static final ImmutableList<String> BAD_SUFFIXES =
      ImmutableList.of("-long-repeat.sl", "-long.sl", "-short.sl", "_short.sl", "_small.sl");

  private static String stripQuotes(String s) {
    // This is quite inefficient but this part doesn't contribute to synthesis time.
    while (s.startsWith("\"")) {
      s = s.substring(1);
    }
    while (s.endsWith("\"")) {
      s = s.substring(0, s.length() - 1);
    }
    return s;
  }

  private static Optional<Benchmark> processSygusFile(File file) throws IOException {
    String name = file.getName();
    if (!name.endsWith(SL_EXTENSION)) {
      return Optional.empty();
    }
    for (String badSuffix : BAD_SUFFIXES) {
      if (name.endsWith(badSuffix)) {
        return Optional.empty();
      }
    }

    String content = Files.asCharSource(file, StandardCharsets.UTF_8).read();
    content = content.replaceAll("\\s+", " ");

    List<Example> examples = new ArrayList<>();
    Matcher matcher = Pattern.compile("\\(constraint \\(= \\(f \"(.*?)\"\\)\\)").matcher(content);
    while (matcher.find()) {
      String match = matcher.group(1);
      List<String> split = Splitter.on("\")").splitToList(match);
      if (split.size() != 2) {
        return Optional.empty();
      }
      String[] inputs = stripQuotes(split.get(0).trim()).split("\"\\s+\"");
      String output = stripQuotes(split.get(1).trim());
      examples.add(Example.create(Arrays.asList(inputs), output));
    }

    if (examples.isEmpty()) {
      return Optional.empty();
    }

    String description = "SyGuS benchmark from " + file.getPath();
    String expectedProgram = "";
    Benchmark benchmark =
        Benchmark.createBenchmarkWithNoTags(name, description, examples, examples, expectedProgram);
    return Optional.of(benchmark);
  }

  public static List<Benchmark> getSygusBenchmarks() throws IOException {
    Queue<File> files = new ArrayDeque<>();
    String path = sygusBenchmarksDirectory.replaceFirst("^~", System.getProperty("user.home"));
    files.add(new File(path));
    List<Benchmark> benchmarks = new ArrayList<>();
    while (!files.isEmpty()) {
      File file = files.poll();
      if (file.getName().startsWith(".")) { // Ignore hidden directories and files.
        continue;
      }
      if (file.isDirectory()) {
        File[] contents = file.listFiles();
        Arrays.sort(contents, Comparator.comparing(File::getName));
        Collections.addAll(files, contents);
      } else {
        Optional<Benchmark> maybeBenchmark = processSygusFile(file);
        if (maybeBenchmark.isPresent()) {
          benchmarks.add(maybeBenchmark.get());
        }
      }
    }
    return benchmarks;
  }

  private SygusBenchmarks() {}
}
