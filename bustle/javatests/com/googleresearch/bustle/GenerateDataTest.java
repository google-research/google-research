package com.googleresearch.bustle;

import static com.google.common.truth.Truth.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import picocli.CommandLine;

@RunWith(JUnit4.class)
public class GenerateDataTest {

  @Test
  public void checkDeterminism() throws Exception {
    Synthesizer synthesizer = new Synthesizer();
    CommandLine cmd = new CommandLine(synthesizer);
    int exitCode = cmd.execute("--max_expressions=50000", "--parse_flags_only=True");
    assertThat(exitCode).isEqualTo(0);
    assertThat((List<?>) cmd.getExecutionResult()).isEmpty(); // Only parsed flags.

    int numSearches = 3;
    int numValuesPerSearch = 1000;
    int numOverallRuns = 4;
    List<List<String>> resultLists = new ArrayList<>();
    for (int i = 0; i < numOverallRuns; i++) {
      Random randomGen = new Random();
      randomGen.setSeed(0); // Set the seed for reproducible generation
      List<List<DataItem>> data =
          GenerateData.generateData(numSearches, numValuesPerSearch, randomGen);
      List<TrainingDataItem> trainingDataItems =
          GenerateData.buildTrainingDataItems(data, randomGen);
      List<String> subExpressions = new ArrayList<>();
      for (TrainingDataItem tde : trainingDataItems) {
        subExpressions.add(tde.getSubExpression().expression());
      }
      resultLists.add(subExpressions);
    }
    for (int i = 1; i < numOverallRuns; i++) {
      assertThat(resultLists.get(i)).containsExactlyElementsIn(resultLists.get(0));
    }
  }
}
