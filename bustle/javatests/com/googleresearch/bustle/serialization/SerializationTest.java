package com.googleresearch.bustle.serialization;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import com.googleresearch.bustle.Operation;
import com.googleresearch.bustle.propertysignatures.ComputeSignature;
import com.googleresearch.bustle.propertysignatures.PropertySummary;
import com.googleresearch.bustle.value.ConstantValue;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.OperationValue;
import com.googleresearch.bustle.value.Value;
import java.lang.reflect.Method;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class SerializationTest {

  @Test
  public void checkStringInputValueRoundTrip() throws Exception {
    InputValue inputValue = new InputValue(Arrays.asList("butter", "bunter", "xxy"), "inputs");
    Gson gson = SerializationUtils.constructCustomGsonBuilder();
    String inputValueJson = gson.toJson(inputValue);
    InputValue parsedInputValue = gson.fromJson(inputValueJson, InputValue.class);
    assertThat(parsedInputValue.equals(inputValue)).isTrue();
  }

  @Test
  public void checkIntInputValueRoundTrip() throws Exception {
    InputValue inputValue = new InputValue(Arrays.asList(1, 2, 3), "inputs");
    Gson gson = SerializationUtils.constructCustomGsonBuilder();
    String inputValueJson = gson.toJson(inputValue);
    InputValue parsedInputValue = gson.fromJson(inputValueJson, InputValue.class);
    assertThat(parsedInputValue.equals(inputValue)).isTrue();
  }

  @Test
  public void checkIntConstantValueRoundTrip() throws Exception {
    ConstantValue constantValue = new ConstantValue(42, 3);
    Gson gson = SerializationUtils.constructCustomGsonBuilder();
    String constantValueJson = gson.toJson(constantValue);
    ConstantValue parsedConstantValue = gson.fromJson(constantValueJson, ConstantValue.class);
    assertThat(parsedConstantValue.equals(constantValue)).isTrue();
  }

  @Test
  public void checkStringConstantValueRoundTrip() throws Exception {
    ConstantValue constantValue = new ConstantValue("foo", 7);
    Gson gson = SerializationUtils.constructCustomGsonBuilder();
    String constantValueJson = gson.toJson(constantValue);
    ConstantValue parsedConstantValue = gson.fromJson(constantValueJson, ConstantValue.class);
    assertThat(parsedConstantValue.equals(constantValue)).isTrue();
  }

  @Test
  public void checkOperationValueRoundTrip() throws Exception {
    Operation leftOp = Operation.lookupOperation("LEFT", 2);

    // Create the inputs we'll pass to LEFT, then pass them to LEFT to create an opValue
    InputValue findTxt = new InputValue(Arrays.asList("abcde", "fghij", "klmno"), "inputs");
    ConstantValue numChars = new ConstantValue(4, 3);
    List<Value> args = Arrays.asList(findTxt, numChars);
    Value operationValue = leftOp.apply(args);

    // Serialize and then deserialize the opValue
    Gson gson = SerializationUtils.constructCustomGsonBuilder();
    String operationValueJson = gson.toJson(operationValue, OperationValue.class);
    OperationValue parsedOperationValue = gson.fromJson(operationValueJson, OperationValue.class);

    // this only checks for equality of the wrappedValues
    assertThat(parsedOperationValue.equals(operationValue)).isTrue();

    // now check that the Operation made a round trip correctly
    // We check this by checking the Method, since the Method uniquely determines the Op
    Method parsedMethod = parsedOperationValue.getOperation().getMethod();
    Operation rightOp = Operation.lookupOperation("RIGHT", 2);
    assertThat(parsedMethod.equals(leftOp.getMethod())).isTrue();
    assertThat(parsedMethod.equals(rightOp.getMethod())).isFalse();

    // Test that the internal nodes (from arguments) were correctly converted to Values
    List<Value> parsedArgs = parsedOperationValue.getArguments();
    assertThat(parsedArgs.get(0)).isInstanceOf(InputValue.class);
    assertThat(parsedArgs.get(1)).isInstanceOf(ConstantValue.class);
  }

  @Test
  public void checkValuesRoundTrip() throws Exception {
    // Check that deserializing as Value will yield the correct Value sub-class
    Operation leftOp = Operation.lookupOperation("LEFT", 2);

    // Create the inputs we'll pass to LEFT, then pass them to LEFT to create an opValue
    InputValue inputValue = new InputValue(Arrays.asList("abcde", "fghij", "klmno"), "inputs");
    ConstantValue constantValue = new ConstantValue(4, 3);
    List<Value> args = Arrays.asList(inputValue, constantValue);
    Value operationValue = leftOp.apply(args);

    // Serialize and then deserialize inputValue, operationValue, and constantValue
    Gson gson = SerializationUtils.constructCustomGsonBuilder();

    String operationValueJson = gson.toJson(operationValue, OperationValue.class);
    Value parsedOperationValue = gson.fromJson(operationValueJson, Value.class);

    String inputValueJson = gson.toJson(inputValue, InputValue.class);
    Value parsedInputValue = gson.fromJson(inputValueJson, Value.class);

    String constantValueJson = gson.toJson(constantValue, ConstantValue.class);
    Value parsedConstantValue = gson.fromJson(constantValueJson, Value.class);

    assertThat(parsedOperationValue).isInstanceOf(OperationValue.class);
    assertThat(parsedInputValue).isInstanceOf(InputValue.class);
    assertThat(parsedConstantValue).isInstanceOf(ConstantValue.class);
  }

  @Test
  public void checkPropertySignatureRoundTrip() throws Exception {
    List<InputValue> inputs =
        ImmutableList.of(new InputValue(Arrays.asList("butter", "bunter", "xxy"), "inputs"));
    Value outputValue = new InputValue(Arrays.asList("butterfly", "butterfly", "xx"), "outputs");

    List<PropertySummary> propertySignature =
        ComputeSignature.computeExampleSignature(inputs, outputValue);

    Gson gson = SerializationUtils.constructCustomGsonBuilder();

    String propertySignatureJson = gson.toJson(propertySignature);
    Type listType = new TypeToken<List<PropertySummary>>() {}.getType();
    List<PropertySummary> deserializedSig = gson.fromJson(propertySignatureJson, listType);
    assertThat(deserializedSig.equals(propertySignature)).isTrue();
  }
}
