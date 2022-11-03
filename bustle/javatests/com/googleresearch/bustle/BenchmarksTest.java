package com.googleresearch.bustle;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;


@RunWith(Parameterized.class)
public final class BenchmarksTest {

  @Parameters(name = "{index}...\n{0}")
  public static ImmutableList<Benchmark> data() {
    return Benchmarks.ALL_BENCHMARKS;
  }

  @Parameter public Benchmark benchmark;

  @Test
  public void hasSolution() throws Exception {
    if (benchmark.getName().endsWith("ShouldFail")) {
      assertThat(benchmark.getExpectedProgram()).isEmpty();
    } else {
      assertThat(benchmark.getExpectedProgram()).isNotEmpty();
    }
  }
}
