package com.googleresearch.bustle;

import static java.lang.Math.max;
import static java.lang.Math.min;

import com.google.common.base.Ascii;

/**
 * Implements supported Sheets functions.
 */
public final class SheetsFunctions {

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static Integer add(Integer a, Integer b) {
    return a + b;
  }

  static String concatenate(String a, String b) {
    return a + b;
  }

  // static String concatenate(String a, String b, String c) {
  //   return a + b + c;
  // }

  static Integer find(String findTxt, String withinTxt) {
    return find(findTxt, withinTxt, 1);
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static Integer find(String findTxt, String withinTxt, Integer startPos) {
    if (startPos <= 0) {
      return null;
    }
    int result = withinTxt.indexOf(findTxt, startPos - 1);
    if (result == -1) {
      return null;
    }
    return result + 1;
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static String left(String txt, Integer numChars) {
    if (numChars < 0) {
      return null;
    } else if (numChars > txt.length()) {
      return txt;
    } else {
      return txt.substring(0, numChars);
    }
  }

  static Integer len(String txt) {
    return txt.length();
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static String mid(String txt, Integer startPosition, Integer numChars) {
    if (startPosition <= 0 || numChars < 0) {
      return null;
    }
    int startIndex = startPosition - 1;
    if (startIndex > txt.length()) {
      return "";
    }
    int endIndex = startPosition - 1 + numChars;
    if (endIndex > txt.length()) {
      return txt.substring(startIndex);
    } else {
      return txt.substring(startIndex, endIndex);
    }
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static Integer minus(Integer a, Integer b) {
    return a - b;
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static String replace(String text, Integer startPos, Integer numChars, String newText) {
    if (startPos <= 0 || numChars < 0) {
      return null;
    }
    int startIndex = min(startPos - 1, text.length());
    return text.substring(0, startIndex)
        + newText
        + text.substring(min(text.length(), startIndex + numChars));
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static String right(String txt, Integer numChars) {
    if (numChars < 0) {
      return null;
    }
    return txt.substring(max(0, txt.length() - numChars));
  }

  static String trim(String text) {
    return text.trim().replaceAll("  +", " ");
  }

  static String lower(String text) {
    return Ascii.toLowerCase(text);
  }

  static String upper(String text) {
    return Ascii.toUpperCase(text);
  }

  static String proper(String text) {
    text = Ascii.toLowerCase(text);
    StringBuilder builder = new StringBuilder();
    boolean prevWasLetter = false;
    // Capitalize all letters that follow a non-letter.
    for (int i = 0; i < text.length(); i++) {
      char c = text.charAt(i);
      boolean isLetter = Character.isLetter(c);
      if (isLetter && !prevWasLetter) {
        c = Character.toUpperCase(c);
      }
      builder.append(c);
      prevWasLetter = isLetter;
    }
    return builder.toString();
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static String rept(String str, Integer intTimes) {
    if (intTimes < 0 || str.length() * intTimes > 100) {
      return null;
    }
    return str.repeat(intTimes);
  }

  static String substitute(String source, String search, String replace) {
    return substitute(source, search, replace, 0);
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static String substitute(String source, String search, String replace, Integer occurrence) {
    if (occurrence < 0) {
      return null;
    }
    if (search.isEmpty()) {
      return source;
    }
    if (occurrence == 0) {
      return source.replace(search, replace);
    }
    int index = -1;
    for (int i = 0; i < occurrence; i++) {
      index++;
      index = source.indexOf(search, index);
      if (index == -1) {
        break;
      }
    }
    if (index != -1) {
      return source.substring(0, index) + replace + source.substring(index + search.length());
    } else {
      return source;
    }
  }

  static String to_text(Integer number) {
    return "" + number;
  }

  // The strange capitalization of `iF` is so that:
  // 1. its uppercased form, `IF`, is exactly the Sheets function we want to call
  // 2. it's not entirely lowercase, because `if` is a keyword
  // 3. it doesn't start uppercase, because that would anger the linter which can't be ignored
  @SuppressWarnings("UnnecessaryBoxedVariable")
  static String iF(Boolean condition, String resultIfTrue, String resultIfFalse) {
    return condition ? resultIfTrue : resultIfFalse;
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static Boolean exact(String left, String right) {
    return left.equals(right);
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static Boolean gt(Integer left, Integer right) {
    return left > right;
  }

  @SuppressWarnings("UnnecessaryBoxedVariable")
  static Boolean gte(Integer left, Integer right) {
    return left >= right;
  }

  private SheetsFunctions() {}
}
