package com.googleresearch.bustle;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import java.lang.reflect.Method;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.junit.runners.Parameterized.Parameter;
import org.junit.runners.Parameterized.Parameters;

@RunWith(Parameterized.class)
public final class SheetsFunctionsTest {

  @Parameters(name = "{0}({1}) -> {2}")
  public static List<Object[]> data() {
    return Arrays.asList(
        new Object[][] {
          {"add", ImmutableList.of(3, 5), 8},
          {"concatenate", ImmutableList.of("a", "bc"), "abc"},
          {"find", ImmutableList.of("b", "abc"), 2},
          {"find", ImmutableList.of("B", "abc"), null},
          {"find", ImmutableList.of("b", "abcb", 0), null},
          {"find", ImmutableList.of("b", "abcb", 1), 2},
          {"find", ImmutableList.of("b", "abcb", 2), 2},
          {"find", ImmutableList.of("b", "abcb", 3), 4},
          {"find", ImmutableList.of("b", "abcb", 4), 4},
          {"find", ImmutableList.of("b", "abcb", 5), null},
          {"find", ImmutableList.of("B", "abcb", 2), null},
          {"left", ImmutableList.of("abc", -1), null},
          {"left", ImmutableList.of("abc", 0), ""},
          {"left", ImmutableList.of("abc", 2), "ab"},
          {"left", ImmutableList.of("abc", 3), "abc"},
          {"left", ImmutableList.of("abc", 4), "abc"},
          {"left", ImmutableList.of("", 2), ""},
          {"len", ImmutableList.of(""), 0},
          {"len", ImmutableList.of("abc"), 3},
          {"mid", ImmutableList.of("abc", 0, 1), null},
          {"mid", ImmutableList.of("abc", 1, -1), null},
          {"mid", ImmutableList.of("abc", 1, 0), ""},
          {"mid", ImmutableList.of("abc", 1, 1), "a"},
          {"mid", ImmutableList.of("abc", 1, 9), "abc"},
          {"mid", ImmutableList.of("abc", 2, 0), ""},
          {"mid", ImmutableList.of("abc", 2, 1), "b"},
          {"mid", ImmutableList.of("abc", 2, 2), "bc"},
          {"mid", ImmutableList.of("abc", 3, 2), "c"},
          {"mid", ImmutableList.of("abc", 9, 9), ""},
          {"replace", ImmutableList.of("abc", 0, 1, "XY"), null},
          {"replace", ImmutableList.of("abc", 1, -1, "XY"), null},
          {"replace", ImmutableList.of("abc", 1, 0, "XY"), "XYabc"},
          {"replace", ImmutableList.of("abc", 1, 1, "XY"), "XYbc"},
          {"replace", ImmutableList.of("abc", 1, 3, "XY"), "XY"},
          {"replace", ImmutableList.of("abc", 1, 4, "XY"), "XY"},
          {"replace", ImmutableList.of("abc", 2, 1, "XY"), "aXYc"},
          {"replace", ImmutableList.of("abc", 3, 0, "XY"), "abXYc"},
          {"replace", ImmutableList.of("abc", 4, 0, "XY"), "abcXY"},
          {"replace", ImmutableList.of("abc", 5, 0, "XY"), "abcXY"},
          {"replace", ImmutableList.of("abc", 9, 9, "XY"), "abcXY"},
          {"right", ImmutableList.of("abc", -1), null},
          {"right", ImmutableList.of("abc", 0), ""},
          {"right", ImmutableList.of("abc", 1), "c"},
          {"right", ImmutableList.of("abc", 2), "bc"},
          {"right", ImmutableList.of("abc", 3), "abc"},
          {"right", ImmutableList.of("abc", 4), "abc"},
          {"right", ImmutableList.of("", 2), ""},
          {"trim", ImmutableList.of(""), ""},
          {"trim", ImmutableList.of("   "), ""},
          {"trim", ImmutableList.of("    a    "), "a"},
          {"trim", ImmutableList.of("abc"), "abc"},
          {"trim", ImmutableList.of(" abc"), "abc"},
          {"trim", ImmutableList.of("abc "), "abc"},
          {"trim", ImmutableList.of(" abc "), "abc"},
          {"trim", ImmutableList.of("    a   b    c  "), "a b c"},
          {"lower", ImmutableList.of("abc dEf XYZ 123"), "abc def xyz 123"},
          {"upper", ImmutableList.of("abc dEf XYZ 123"), "ABC DEF XYZ 123"},
          {"proper", ImmutableList.of("abc dEf XYZ 123"), "Abc Def Xyz 123"},
          {"proper", ImmutableList.of("a 1a a1 a1a1a .a?a"), "A 1A A1 A1A1A .A?A"},
          {"rept", ImmutableList.of("abc", -1), null},
          {"rept", ImmutableList.of("abc", 0), ""},
          {"rept", ImmutableList.of("abc", 1), "abc"},
          {"rept", ImmutableList.of("abc", 2), "abcabc"},
          {"rept", ImmutableList.of("abc", 3), "abcabcabc"},
          {"substitute", ImmutableList.of("Spreadsheet", "e", "E"), "SprEadshEEt"},
          {"substitute", ImmutableList.of("Spreadsheet", "x", "E"), "Spreadsheet"},
          {"substitute", ImmutableList.of("Spreadsheet", "", "X"), "Spreadsheet"},
          {"substitute", ImmutableList.of("", "", "X"), ""},
          {"substitute", ImmutableList.of("Spreadsheet", "e", ""), "Spradsht"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "X"), "XXA"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "AXA"), "AXAAXAA"},
          {"substitute", ImmutableList.of("AAAAA", "", "B"), "AAAAA"},
          {"substitute", ImmutableList.of("Google Docs", "ogle", "od", 1), "Good Docs"},
          {"substitute", ImmutableList.of("Google Docs", "o", "a", 3), "Google Dacs"},
          {"substitute", ImmutableList.of("Spreadsheet", "e", "E", 2), "SpreadshEet"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "X", -1), null},
          {"substitute", ImmutableList.of("AAAAA", "AA", "X", 0), "XXA"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "X", 1), "XAAA"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "X", 2), "AXAA"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "X", 3), "AAXA"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "X", 4), "AAAX"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "X", 5), "AAAAA"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "AXA", 0), "AXAAXAA"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "AXA", 1), "AXAAAA"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "AXA", 4), "AAAAXA"},
          {"substitute", ImmutableList.of("AAAAA", "AA", "AXA", 5), "AAAAA"},
          {"substitute", ImmutableList.of("A", "", "B", -1), null},
          {"to_text", ImmutableList.of(-99), "-99"},
          {"to_text", ImmutableList.of(0), "0"},
          {"to_text", ImmutableList.of(123), "123"},
          {"iF", ImmutableList.of(true, "A", "B"), "A"},
          {"iF", ImmutableList.of(false, "A", "B"), "B"},
          {"exact", ImmutableList.of("A", "A"), true},
          {"exact", ImmutableList.of("A", "B"), false},
          {"gt", ImmutableList.of(5, 6), false},
          {"gt", ImmutableList.of(6, 6), false},
          {"gt", ImmutableList.of(7, 6), true},
          {"gte", ImmutableList.of(5, 6), false},
          {"gte", ImmutableList.of(6, 6), true},
          {"gte", ImmutableList.of(7, 6), true},
        });
  }

  @Parameter(0)
  public String methodName;

  @Parameter(1)
  public ImmutableList<?> args;

  @Parameter(2)
  public Object expectedResult;

  @Test
  public void checkFunction() throws Exception {
    Method[] allMethods = SheetsFunctions.class.getDeclaredMethods();
    int arity = args.size();
    Method desiredMethod = null;
    for (Method m : allMethods) {
      if (m.getName().equals(methodName) && m.getParameterCount() == arity) {
        desiredMethod = m;
        break;
      }
    }
    assertThat(desiredMethod).isNotNull();
    assertThat(desiredMethod.invoke(null, args.toArray())).isEqualTo(expectedResult);
  }
}
