package com.googleresearch.bustle;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.Comparator.naturalOrder;

import com.googleresearch.bustle.value.InputValue;
import com.googleresearch.bustle.value.OutputValue;
import com.googleresearch.bustle.value.Value;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Heuristics for extracting constants. These need to be fast, since they have to run before the
 * synthesizer does.
 */
public class ConstantExtraction {

  private ConstantExtraction() {}

  /**
   * Finds the longest common substring between two strings.
   *
   * <p>The long common substring of 123456 and 1356234 is 234. (The LCSubsequence would be 1356 or
   * 1234)
   */
  public static String computeLongestCommonSubstring(String str1, String str2) {
    int len1 = str1.length();
    int len2 = str2.length();

    // opt[i][j] = length of LCSubstring of x[i..M] and y[j..N]
    int[][] optimalSubstringLength = new int[len1 + 1][len2 + 1];

    String bestSubstringSoFar = "";
    int bestLengthSoFar = 0;
    for (int i = 1; i <= len1; i++) {
      for (int j = 1; j <= len2; j++) {
        if (str1.charAt(i - 1) == str2.charAt(j - 1)) {
          optimalSubstringLength[i][j] = optimalSubstringLength[i - 1][j - 1] + 1;
          if (optimalSubstringLength[i][j] > bestLengthSoFar) {
            bestLengthSoFar = optimalSubstringLength[i][j];
            bestSubstringSoFar = str1.substring(i - bestLengthSoFar, i);
          }
        }
      }
    }

    return bestSubstringSoFar;
  }

  /**
   * Extracts constants from the input-output pairs.
   *
   * Extracted constants include:
   *  1. The longest common susbtring between any pair of output strings, or any pair of input
   *     strings from the same column, if it has >= 2 characters.
   *  2. Any string constant from the provided list of common string constants, if it is a substring
   *     of any input or output.
   */
  public static List<String> extractConstants(
      List<InputValue> inputValues, OutputValue outputValue, List<String> commonConstants) {

    Set<String> extractedConstantsSet = new HashSet<>();

    List<List<String>> stringColumns = new ArrayList<>();
    List<Value> values = new ArrayList<>(inputValues);
    values.add(outputValue);
    for (Value value : values) {
      if (value.getType().equals(String.class)) {
        stringColumns.add(
            value.getWrappedValues().stream().map(o -> (String) o).collect(toImmutableList()));
      }
    }

    // Criteria 1: longest common substrings.
    List<Pair<String, String>> lcsPairs = new ArrayList<>();
    for (List<String> stringColumn : stringColumns) {
      for (int i = 0; i < stringColumn.size(); i++) {
        for (int j = i + 1; j < stringColumn.size(); j++) {
          lcsPairs.add(new Pair<>(stringColumn.get(i), stringColumn.get(j)));
        }
      }
    }
    for (Pair<String, String> lcsPair : lcsPairs) {
      String lcs = computeLongestCommonSubstring(lcsPair.getFirst(), lcsPair.getSecond());
      if (lcs.length() >= 2) {
        extractedConstantsSet.add(lcs);
      }
    }

    // Criteria 2: common string constants.
    for (String common : commonConstants) {
      boolean use = false;
      columnLoop:
      for (List<String> stringColumn : stringColumns) {
        for (String s : stringColumn) {
          if (s.contains(common)) {
            use = true;
            break columnLoop;
          }
        }
      }
      if (use) {
        extractedConstantsSet.add(common);
      }
    }

    // Sort the constants by length and then alphabetically.
    List<String> extractedConstants = new ArrayList<>(extractedConstantsSet);
    Collections.sort(
        extractedConstants, Comparator.comparingInt(String::length).thenComparing(naturalOrder()));
    return extractedConstants;
  }
}
