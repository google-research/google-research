package com.googleresearch.bustle.propertysignatures;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.OutputValue;
import com.googleresearch.bustle.value.Value;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ComputeSignatureTest {

  @Test
  public void computeExampleSignatureTest() throws Exception {
    // Test examples
    ImmutableList<InputValue> inputs =
        ImmutableList.of(new InputValue(Arrays.asList("butter", "abc", "xyz"), "input_1"));
    OutputValue outputValue = new OutputValue(Arrays.asList("butterfly", "abc_", "XYZ_"));

    List<PropertySummary> propertySignature =
        ComputeSignature.computeExampleSignature(inputs, outputValue);

    // String properties of the input
    assertThat(propertySignature.subList(0, 5))
        .containsExactly(
            PropertySummary.ALL_FALSE, // is empty?
            PropertySummary.ALL_FALSE, // is single char?
            PropertySummary.MIXED, // is short string?
            PropertySummary.ALL_TRUE, // is lowercase?
            PropertySummary.ALL_FALSE // is uppercase?
            );

    // Int properties of the input
    int startIndex = ComputeSignature.NUM_STRING_PROPERTIES;
    assertThat(
            propertySignature.subList(startIndex, startIndex + ComputeSignature.NUM_INT_PROPERTIES))
        .isEqualTo(
            Collections.nCopies(
                ComputeSignature.NUM_INT_PROPERTIES, PropertySummary.TYPE_MISMATCH));

    // String properties of the input compared to the output
    startIndex = ComputeSignature.NUM_SINGLE_VALUE_PROPERTIES;
    assertThat(propertySignature.subList(startIndex, startIndex + 9))
        .containsExactly(
            PropertySummary.MIXED, // output contains input?
            PropertySummary.MIXED, // output starts with input?
            PropertySummary.ALL_FALSE, // output ends with input?
            PropertySummary.ALL_FALSE, // input contains output?
            PropertySummary.ALL_FALSE, // input starts with output?
            PropertySummary.ALL_FALSE, // input ends with output?
            PropertySummary.ALL_TRUE, // output contains input ignoring case?
            PropertySummary.ALL_TRUE, // output starts with input ignoring case?
            PropertySummary.ALL_FALSE // output ends with input ignoring case?
            )
        .inOrder();

    // Int properties of the input compared to the output
    startIndex =
        ComputeSignature.NUM_SINGLE_VALUE_PROPERTIES
            + ComputeSignature.NUM_STRING_COMPARISON_PROPERTIES;
    assertThat(
            propertySignature.subList(
                startIndex, startIndex + ComputeSignature.NUM_INT_COMPARISON_PROPERTIES))
        .isEqualTo(
            Collections.nCopies(
                ComputeSignature.NUM_INT_COMPARISON_PROPERTIES, PropertySummary.TYPE_MISMATCH));

    // Inputs 2 and 3 are "padding"
    startIndex =
        ComputeSignature.NUM_SINGLE_VALUE_PROPERTIES + ComputeSignature.NUM_COMPARISON_PROPERTIES;
    assertThat(propertySignature.subList(startIndex, 3 * startIndex))
        .isEqualTo(Collections.nCopies(2 * startIndex, PropertySummary.TYPE_MISMATCH));

    // String properties of the output
    startIndex = 3 * startIndex;
    assertThat(propertySignature.subList(startIndex, startIndex + 5))
        .containsExactly(
            PropertySummary.ALL_FALSE, // is empty?
            PropertySummary.ALL_FALSE, // is single char?
            PropertySummary.MIXED, // is short string?
            PropertySummary.MIXED, // is lowercase?
            PropertySummary.MIXED // is uppercase?
            )
        .inOrder();

    // Integer properties of the output
    startIndex += ComputeSignature.NUM_STRING_PROPERTIES;
    assertThat(
            propertySignature.subList(startIndex, startIndex + ComputeSignature.NUM_INT_PROPERTIES))
        .isEqualTo(
            Collections.nCopies(
                ComputeSignature.NUM_INT_PROPERTIES, PropertySummary.TYPE_MISMATCH));

    assertThat(propertySignature)
        .hasSize(
            (ComputeSignature.MAX_INPUTS + 1) * ComputeSignature.NUM_SINGLE_VALUE_PROPERTIES
                + ComputeSignature.MAX_INPUTS * ComputeSignature.NUM_COMPARISON_PROPERTIES);
  }

  @Test
  public void computeValueSignatureTest() throws Exception {
    // Test examples
    Value value = new InputValue(Arrays.asList("butter", "abc", "xyz"), "inputs");
    Value outputValue = new InputValue(Arrays.asList("butterfly", "abc_", "XYZ_"), "outputs");

    List<PropertySummary> valueSignature =
        ComputeSignature.computeValueSignature(value, outputValue);
    List<PropertySummary> exampleSignature =
        ComputeSignature.computeExampleSignature(ImmutableList.of((InputValue) value), outputValue);

    // valueSignature should a prefix of exampleSignature
    int expectedLength =
        ComputeSignature.NUM_STRING_PROPERTIES
            + ComputeSignature.NUM_INT_PROPERTIES
            + ComputeSignature.NUM_BOOL_PROPERTIES
            + ComputeSignature.NUM_STRING_COMPARISON_PROPERTIES
            + ComputeSignature.NUM_INT_COMPARISON_PROPERTIES;
    assertThat(valueSignature).isEqualTo(exampleSignature.subList(0, expectedLength));
  }
}
