package com.googleresearch.bustle;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.googleresearch.bustle.exception.SynthesisError;
import com.googleresearch.bustle.value.ConstantValue;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.Value;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class UtilsTest {

  @Test
  public void generatePartitionsReturnsExpected() throws Exception {
    assertThat(Utils.generatePartitions(5, 3))
        .containsExactly(
            Arrays.asList(1, 1, 3),
            Arrays.asList(1, 2, 2),
            Arrays.asList(1, 3, 1),
            Arrays.asList(2, 1, 2),
            Arrays.asList(2, 2, 1),
            Arrays.asList(3, 1, 1));

    assertThat(Utils.generatePartitions(5, 1)).containsExactly(Arrays.asList(5));
    assertThat(Utils.generatePartitions(5, 5)).containsExactly(Arrays.asList(1, 1, 1, 1, 1));
    assertThat(Utils.generatePartitions(5, 6)).isEmpty();
    assertThat(Utils.generatePartitions(0, 5)).isEmpty();
  }

  @Test
  public void generatePartitionsRaises() throws Exception {
    assertThrows(SynthesisError.class, () -> Utils.generatePartitions(-1, 3));
    assertThrows(SynthesisError.class, () -> Utils.generatePartitions(5, 0));
  }

  @Test
  public void checkSubExpressionGetter() throws Exception {
    // create the following expression:
    // CONCATENATE(LEFT(in_1, 4), in_2)
    Operation leftOp = Operation.lookupOperation("LEFT", 2);
    Operation concatOp = Operation.lookupOperation("CONCATENATE", 2);

    InputValue findTxt = new InputValue(Arrays.asList("abcde", "fghij", "klmno"), "in_1");
    ConstantValue numChars = new ConstantValue(4, 3);
    List<Value> leftArgs = Arrays.asList(findTxt, numChars);
    Value leftValue = leftOp.apply(leftArgs);

    InputValue muppetTxt = new InputValue(Arrays.asList("_kermit", "_animal", "_beaker"), "in_2");
    List<Value> concatArgs = Arrays.asList(leftValue, muppetTxt);
    Value concatValue = concatOp.apply(concatArgs);

    Set<Value> subExpressions = Utils.getSubExpressions(concatValue);
    assertThat(subExpressions)
        .containsExactly(
            findTxt, // in_1
            numChars, // 4
            leftValue, // LEFT(in_1, 4)
            muppetTxt, // in_2
            concatValue // CONCATENATE(LEFT(in_1, 4), in_2)
            );
  }
}
