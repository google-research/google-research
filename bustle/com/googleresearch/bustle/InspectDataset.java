package com.googleresearch.bustle;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.gson.Gson;
import com.googleresearch.bustle.propertysignatures.PropertySummary;
import com.googleresearch.bustle.serialization.SerializationUtils;
import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.Value;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/** A utility to inspect a generated synthetic dataset. */
@Command(name = "InspectDataset", mixinStandardHelpOptions = true)
public final class InspectDataset implements Runnable {

  @SuppressWarnings("FieldCanBeFinal") // picocli modifies value when flag is set
  @Option(
      names = {"--input_file"},
      description = "Where the training data is stored.")
  private static String inputFile = "/tmp/training_data.json";

  /** Process inputs saved out by GenerateData class. */
  public static List<TrainingDataItem> extract() throws IOException {
    System.out.println("Extracting ...");
    Gson gson = SerializationUtils.constructCustomGsonBuilder();
    List<TrainingDataItem> deserializedValues = new ArrayList<>();
    try {
      List<String> jsonStrings = Files.readAllLines(Paths.get(inputFile), UTF_8);
      for (String s : jsonStrings) {
        TrainingDataItem parsedVal = gson.fromJson(s, TrainingDataItem.class);
        deserializedValues.add(parsedVal);
      }
    } catch (IOException e) {
      System.out.println("We could not deserialize the data.");
      throw e;
    }
    return deserializedValues;
  }

  @Override
  public void run() {
    List<TrainingDataItem> deserializedValues;
    try {
      deserializedValues = extract();
    } catch (IOException e) {
      System.err.println(e);
      return;
    }
    for (TrainingDataItem v : deserializedValues) {
      List<InputValue> inputValues = v.getInputValues();
      Value subExpression = v.getSubExpression();
      Value targetExpression = v.getTargetExpression();
      boolean isPositiveExample = v.getIsPositiveExample();
      List<PropertySummary> exampleSignature = v.getExampleSignature();
      List<PropertySummary> valueSignature = v.getValueSignature();
      System.out.println("===========================================================");
      System.out.println("InputValues: ");
      for (InputValue iv : inputValues) {
        System.out.println("  " + iv.expression());
      }
      System.out.println("target expression: ");
      System.out.println("  " + targetExpression.expression());
      System.out.println("sub-expression: ");
      System.out.println("  " + subExpression.expression());
      System.out.println("isPositiveExample: ");
      System.out.println("  " + isPositiveExample);
      System.out.println("exampleSignature: ");
      System.out.println("  " + exampleSignature);
      System.out.println("valueSignature ");
      System.out.println("  " + valueSignature);
    }
  }

  public static void main(String[] args) {
    int exitCode = new CommandLine(new InspectDataset()).execute(args);
    System.exit(exitCode);
  }

  private InspectDataset() {}
}
