package com.googleresearch.bustle;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.OutputValue;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ConstantExtractionTest {

  @Test
  public void computeLongestCommonSubstringReturnsExpected() throws Exception {
    assertThat(ConstantExtraction.computeLongestCommonSubstring("123456", "1356234"))
        .isEqualTo("234");
    assertThat(ConstantExtraction.computeLongestCommonSubstring("aaaaa", "a"))
        .isEqualTo("a");
    assertThat(ConstantExtraction.computeLongestCommonSubstring("exact match", "exact match"))
        .isEqualTo("exact match");
    assertThat(ConstantExtraction.computeLongestCommonSubstring("aaaaa", "b")).isEmpty();
    assertThat(ConstantExtraction.computeLongestCommonSubstring("", "")).isEmpty();
  }

  @Test
  public void extractConstantsReturnsExpected() throws Exception {
    ImmutableList<InputValue> inputs =
        ImmutableList.of(new InputValue(Arrays.asList("abcd", "wxyz", "0ab12"), "input_1"),
            new InputValue(Arrays.asList("$1234", "$5.67", "$89"), "input_2"));
    OutputValue output = new OutputValue(Arrays.asList("01/01/01", "11/30/98", ""));
    List<String> commonConstants = Arrays.asList(
        ".", // Appears once for input_2.
        "#", // Doesn't appear anywhere.
        "/"); // Appears in the output as a substring of another constant, but included anyway.

    assertThat(ConstantExtraction.extractConstants(inputs, output, commonConstants))
        .containsExactly(
            "ab", // LCS from input_1.
            "1/", // LCS from output.
            ".", // Common constant appearing in input_2.
            "/"); // Common constant appearing in output.
    // "$" appears in every instance of input_2, but it is only 1 character, and isn't in the common
    // constants list, so it is not extracted.
  }
}
