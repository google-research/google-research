package com.googleresearch.bustle;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;
import picocli.CommandLine;

@RunWith(Parameterized.class)
public final class SynthesizerTest {

  @Parameters(name = "{0} -> {1}")
  public static ImmutableList<Object[]> data() {
    return ImmutableList.copyOf(
        new Object[][] {
          {"capitalizeSentence", "REPLACE(LOWER(var_0), 1, 1, LEFT(PROPER(var_0), 1))", 9, 5},
          {"dateTransformation2", "SUBSTITUTE(var_0, \"-\", \"/\")", 4, 0.5},
          {"lastNameFirstColumn", "REPLACE(var_0, 1, FIND(\" \", var_0), \"\")", 7, 1},
          {"prependMr", "REPLACE(var_0, 1, FIND(\" \", var_0), \"Mr. \")", 7, 1},
          {"stringEqual", "IF(EXACT(var_0, var_1), \"yes\", \"no\")", 6, 1},
          {"stringLength", "TO_TEXT(LEN(var_0))", 3, 0.5},
        });
  }

  @Parameter(0)
  public String benchmarkName;

  @Parameter(1)
  public String expectedSolution;

  @Parameter(2)
  public int solutionWeight;

  @Parameter(3)
  public double timeLimit;

  @Test
  public void synthesizeBenchmark() throws Exception {
    Synthesizer synthesizer = new Synthesizer();
    CommandLine cmd = new CommandLine(synthesizer);
    int exitCode =
        cmd.execute(
            "--model_reweighting=False",
            "--heuristic_reweighting=False",
            "--premise_selection=False",
            "--parse_flags_only=False",
            "--time_limit=" + timeLimit,
            "--max_expressions=100000000",
            "--benchmark_name=" + benchmarkName);

    assertThat(exitCode).isEqualTo(0);

    List<SynthesisResult> results = cmd.getExecutionResult();
    assertThat(results).hasSize(1);
    SynthesisResult result = results.get(0);
    assertThat(result.getBenchmarkName()).isEqualTo(benchmarkName);
    assertThat(result.getSuccess()).isTrue();
    assertThat(result.getSolution()).isEqualTo(expectedSolution);
    assertThat(result.getSolutionWeight()).isEqualTo(solutionWeight);
    assertThat(result.getElapsedTime()).isLessThan(timeLimit);
  }
}
