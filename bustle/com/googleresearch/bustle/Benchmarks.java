package com.googleresearch.bustle;

import com.google.common.base.Ascii;
import com.google.common.collect.ImmutableList;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** A class holding Benchmarks */
public class Benchmarks {

  public static final ImmutableList<Benchmark> ALL_BENCHMARKS;

  static {
    List<Benchmark> benchmarks = new ArrayList<>();
    Set<String> benchmarkNames = new HashSet<>();

    for (Method method : Benchmarks.class.getDeclaredMethods()) {
      int modifiers = method.getModifiers();
      // Find all methods with the signature `public static Benchmark myMethod() {...}`.
      if (Modifier.isPublic(modifiers)
          && Modifier.isStatic(modifiers)
          && method.getReturnType().equals(Benchmark.class)
          && method.getParameterCount() == 0) {
        try {
          Benchmark benchmark = (Benchmark) method.invoke(null);
          String benchmarkName = benchmark.getName();
          if (!benchmarkName.equals(method.getName())) {
            System.out.println(
                "Warning: method "
                    + method.getName()
                    + " does not match benchmark name "
                    + benchmarkName);
          }
          if (benchmarkNames.contains(benchmarkName)) {
            System.out.println("Warning: duplicate benchmark name " + benchmarkName);
          }
          benchmarks.add(benchmark);
          benchmarkNames.add(benchmarkName);
        } catch (ReflectiveOperationException e) {
          System.out.println("Exception caught in static initializer of Benchmarks: " + e);
        }
      }
    }

    // We iterate over methods in a nondeterministic order. Sort the benchmarks by name to obtain a
    // consistent ordering.
    Collections.sort(benchmarks, Comparator.comparing(Benchmark::getName));

    ALL_BENCHMARKS = ImmutableList.copyOf(benchmarks);
  }

  private Benchmarks() {}

  public static ImmutableList<Benchmark> getBenchmarkWithName(
      String name, List<BenchmarkTag> included, List<BenchmarkTag> excluded) {
    if (Ascii.equalsIgnoreCase(name, "ALL")) {
      List<Benchmark> benchmarks = new ArrayList<>();
      for (Benchmark b : ALL_BENCHMARKS) {
        List<BenchmarkTag> tags = b.getTags();
        Set<BenchmarkTag> includedMatches = new HashSet<>(included);
        includedMatches.retainAll(tags);
        Set<BenchmarkTag> excludedMatches = new HashSet<>(excluded);
        excludedMatches.retainAll(tags);
        if ((included.isEmpty() || !includedMatches.isEmpty()) && excludedMatches.isEmpty()) {
          benchmarks.add(b);
        }
      }
      return ImmutableList.copyOf(benchmarks);
    } else {
      List<Benchmark> matchingBenchmarks = new ArrayList<>();
      for (Benchmark b : ALL_BENCHMARKS) {
        if (b.getName().equals(name)) {
          matchingBenchmarks.add(b);
        }
      }
      return ImmutableList.copyOf(matchingBenchmarks);
    }
  }

  public static ImmutableList<Benchmark> getBenchmarkWithName(String name) {
    return getBenchmarkWithName(name, ImmutableList.of(), ImmutableList.of());
  }

  public static Benchmark prependMr() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("George Washington"), "Mr. Washington"),
            Example.create(Arrays.asList("Alan Turing"), "Mr. Turing"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("A A"), "Mr. A"),
            Example.create(Arrays.asList("A B"), "Mr. B"),
            Example.create(Arrays.asList("Xyz Xyz"), "Mr. Xyz"),
            Example.create(Arrays.asList("Longfirstname X"), "Mr. X"),
            Example.create(Arrays.asList("X Longlastname"), "Mr. Longlastname"),
            Example.create(Arrays.asList("Sundar Pichai"), "Mr. Pichai"));
    String name = "prependMr";
    String description = "prepend Mr. to last name";
    String expectedProgram =
        "CONCATENATE(\"Mr. \", RIGHT(var_0, MINUS(LEN(var_0), FIND(\" \", var_0))))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark prependMrOrMs() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("George Washington", "male"), "Mr. Washington"),
            Example.create(Arrays.asList("Alan Turing", "male"), "Mr. Turing"),
            Example.create(Arrays.asList("Grace Hopper", "female"), "Ms. Hopper"),
            Example.create(Arrays.asList("Ruth Porat", "female"), "Ms. Porat"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("Sundar Pichai", "male"), "Mr. Pichai"),
            Example.create(Arrays.asList("Susan Wojcicki", "female"), "Ms. Wojcicki"),
            Example.create(Arrays.asList("A A", "male"), "Mr. A"),
            Example.create(Arrays.asList("A A", "female"), "Ms. A"),
            Example.create(Arrays.asList("X Longlastname", "male"), "Mr. Longlastname"),
            Example.create(Arrays.asList("X Longlastname", "female"), "Ms. Longlastname"),
            Example.create(Arrays.asList("Longfirstname X", "male"), "Mr. X"),
            Example.create(Arrays.asList("Longfirstname X", "female"), "Ms. X"));
    String name = "prependMrOrMs";
    String description = "prepend Mr. or Ms. to last name depending on gender";
    String expectedProgram =
        "CONCATENATE(IF(EXACT(var_1, \"male\"), \"Mr. \", \"Ms. \"), RIGHT(var_0, MINUS(LEN(var_0),"
            + " FIND(\" \", var_0))))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark appendAmOrPm() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("9:00", "morning"), "9 AM"),
            Example.create(Arrays.asList("11:00", "morning"), "11 AM"),
            Example.create(Arrays.asList("3:00", "afternoon"), "3 PM"),
            Example.create(Arrays.asList("9:00", "night"), "9 PM"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1:00", "morning"), "1 AM"),
            Example.create(Arrays.asList("3:00", "morning"), "3 AM"),
            Example.create(Arrays.asList("10:00", "morning"), "10 AM"),
            Example.create(Arrays.asList("12:00", "morning"), "12 AM"),
            Example.create(Arrays.asList("1:00", "evening"), "1 PM"),
            Example.create(Arrays.asList("3:00", "night"), "3 PM"),
            Example.create(Arrays.asList("10:00", "afternoon"), "10 PM"),
            Example.create(Arrays.asList("12:00", "noon"), "12 PM"));
    String name = "appendAmOrPm";
    String description = "append AM or PM to the hour depending on if it's morning";
    String expectedProgram =
        "CONCATENATE(LEFT(var_0, MINUS(FIND(\":\", var_0), 1)), IF(EXACT(var_1, \"morning\"), \""
            + " AM\", \" PM\"))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark firstNameSecondColumn() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("Larry Page", "Sergey Brin"), "Sergey"),
            Example.create(Arrays.asList("", "Bill Gates"), "Bill"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("Sergey Brin", "Larry Page"), "Larry"),
            Example.create(Arrays.asList("", "George Washington"), "George"),
            Example.create(Arrays.asList("A B", "C D"), "C"),
            Example.create(Arrays.asList("junk", "X Longlastname"), "X"),
            Example.create(Arrays.asList("12345", "Longfirstname X"), "Longfirstname"),
            Example.create(Arrays.asList("very-long-string", "Aa Bb"), "Aa"));
    String name = "firstNameSecondColumn";
    String description = "get first name from second column";
    String expectedProgram = "LEFT(var_1, MINUS(FIND(\" \", var_1), 1))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark lastNameFirstColumn() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("Larry Page", "Sergey Brin"), "Page"),
            Example.create(Arrays.asList("George Washington", ""), "Washington"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("Sergey Brin", "Larry Page"), "Brin"),
            Example.create(Arrays.asList("George Washington", ""), "Washington"),
            Example.create(Arrays.asList("A B", "C D"), "B"),
            Example.create(Arrays.asList("X Longlastname", "junk"), "Longlastname"),
            Example.create(Arrays.asList("Longfirstname X", "12345"), "X"),
            Example.create(Arrays.asList("Aa Bb", "very-long-string"), "Bb"));
    String name = "lastNameFirstColumn";
    String description = "get last name from first column";
    String expectedProgram = "RIGHT(var_0, MINUS(LEN(var_0), FIND(\" \", var_0)))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark createEmailAddress() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("Larry", "Page", "Google"), "lpage@google.com"),
            Example.create(Arrays.asList("Bill", "Gates", "Microsoft"), "bgates@microsoft.com"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("A", "B", "C"), "ab@c.com"),
            Example.create(
                Arrays.asList("Richard", "McDonald", "burgers"), "rmcdonald@burgers.com"),
            Example.create(Arrays.asList("First", "Last", "Company"), "flast@company.com"),
            Example.create(
                Arrays.asList("X", "Longlastname", "Area120"), "xlonglastname@area120.com"),
            Example.create(Arrays.asList("Longfirstname", "X", "Area120"), "lx@area120.com"));
    String name = "createEmailAddress";
    String description = "create email address from name and company";
    String expectedProgram = "LOWER(CONCATENATE(LEFT(var_0, 1), var_1, \"@\", var_2, \".com\"))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark replaceInUrl() {
    List<Example> examples =
        Arrays.asList(
            Example.create(
                Arrays.asList("https://www.google.com/company-strategy.html"),
                "https://www.google.org/company-strategy.html"),
            Example.create(
                Arrays.asList("https://www.google.com/some/path"),
                "https://www.google.org/some/path"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("https://www.google.com/"), "https://www.google.org/"),
            Example.create(
                Arrays.asList("https://www.google.com/com.org.com"),
                "https://www.google.org/com.org.com"),
            Example.create(
                Arrays.asList("https://www.google.com/org/com/"),
                "https://www.google.org/org/com/"),
            Example.create(
                Arrays.asList("https://www.google.com/google.com"),
                "https://www.google.org/google.com"));
    String name = "replaceInUrl";
    String description = "replace com with org";
    String expectedProgram = "SUBSTITUTE(var_0, \"com\", \"org\", 1)";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark replaceAllCompany() {
    List<Example> examples =
        Arrays.asList(
            Example.create(
                Arrays.asList("Employees at <COMPANY> love its culture.", "Google"),
                "Employees at Google love its culture."),
            Example.create(
                Arrays.asList(
                    "<COMPANY> employees are excited for <COMPANY>'s future.", "Microsoft"),
                "Microsoft employees are excited for Microsoft's future."));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("<COMPANY>", "Facebook"), "Facebook"),
            Example.create(Arrays.asList("<COMPANY><COMPANY>", "Area120"), "Area120Area120"),
            Example.create(
                Arrays.asList(" <COMPANY> <COMPANY> <COMPANY> ", "Area120"),
                " Area120 Area120 Area120 "),
            Example.create(Arrays.asList("jeff@<COMPANY>.com", "google"), "jeff@google.com"),
            Example.create(Arrays.asList("jeff@<COMPANY>.com", ""), "jeff@.com"),
            Example.create(
                Arrays.asList("No replacement at all!", "Google"), "No replacement at all!"));
    String name = "replaceAllCompany";
    String description = "replace <COMPANY> in a string with a given company name";
    String expectedProgram = "SUBSTITUTE(var_0, \"<COMPANY>\", var_1)";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark replaceAllNumbersWithPlaceholder() {
    List<Example> examples =
        Arrays.asList(
            Example.create(
                Arrays.asList("Larry sent 339 emails but got only 238 responses."),
                "Larry sent {number} emails but got only {number} responses."),
            Example.create(
                Arrays.asList("Of 238 candidates, we interviewed 100 and made offers to 32."),
                "Of {number} candidates, we interviewed {number} and made offers to {number}."));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("12345678901234567890"), "{number}"),
            Example.create(Arrays.asList("No number anywhere..."), "No number anywhere..."),
            Example.create(
                Arrays.asList("1 2 3 4 5"), "{number} {number} {number} {number} {number}"),
            Example.create(Arrays.asList("127.0.0.1"), "{number}.{number}.{number}.{number}"),
            Example.create(Arrays.asList("1 + 2 = 3"), "{number} + {number} = {number}"),
            Example.create(
                Arrays.asList("I like python3 more than python2."),
                "I like python{number} more than python{number}."));
    String name = "replaceAllNumbersWithPlaceholder";
    String description = "replace all numbers with a placeholder {number}";
    String expectedProgram = "REGEXREPLACE(var_0, \"[0-9]+\", \"{number}\")";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.REGEX);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark extractNumber() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("Larry sent 339 emails today."), "339"),
            Example.create(Arrays.asList("Sergey only sent 45 emails yesterday."), "45"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("99999"), "99999"),
            Example.create(Arrays.asList("python3"), "3"),
            Example.create(Arrays.asList("This sentence has 43 individual characters."), "43"),
            Example.create(Arrays.asList("7890 number in the beginning"), "7890"),
            Example.create(Arrays.asList("number at the end 64862931846"), "64862931846"));
    String name = "extractNumber";
    String description = "extract the single number";
    String expectedProgram = "REGEXEXTRACT(var_0, \"[0-9]+\")";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.REGEX);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark longestCommonPrefix() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("prefix", "present"), "pre"),
            Example.create(Arrays.asList("nonsense", "nonprofit"), "non"),
            Example.create(Arrays.asList("atypical", "asymmetry"), "a"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a", "b"), ""),
            Example.create(Arrays.asList("a", "a"), "a"),
            Example.create(Arrays.asList("word", "word"), "word"),
            Example.create(Arrays.asList("completely", "different"), ""),
            Example.create(Arrays.asList("substring", "substrings"), "substring"),
            Example.create(Arrays.asList("substrings", "substring"), "substring"),
            Example.create(Arrays.asList("side", "insides"), ""),
            Example.create(Arrays.asList("repeatrepeat", "repeatedrepeated"), "repeat"),
            Example.create(Arrays.asList("aaaaa", "aaaaaa"), "aaaaa"),
            Example.create(Arrays.asList("aaaaaa", "aaaaa"), "aaaaa"));
    String name = "longestCommonPrefix";
    String description = "find the longest common prefix of two words";
    String expectedProgram =
        "LEFT(var_0, ARRAYFORMULA(SUM(IF(EQ(LEFT(var_0, SEQUENCE(MIN(LEN(var_0), LEN(var_1)))),"
            + " LEFT(var_1, SEQUENCE(MIN(LEN(var_0), LEN(var_1))))), 1, 0))))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.ARRAY, BenchmarkTag.TOO_DIFFICULT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark longestCommonSuffix() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("suffix", "infix"), "fix"),
            Example.create(Arrays.asList("declaration", "proposition"), "tion"),
            Example.create(Arrays.asList("slyly", "quickly"), "ly"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a", "b"), ""),
            Example.create(Arrays.asList("a", "a"), "a"),
            Example.create(Arrays.asList("word", "word"), "word"),
            Example.create(Arrays.asList("completely", "different"), ""),
            Example.create(Arrays.asList("substring", "asubstring"), "substring"),
            Example.create(Arrays.asList("asubstring", "substring"), "substring"),
            Example.create(Arrays.asList("side", "insides"), ""),
            Example.create(Arrays.asList("repeatrepeat", "repeatedrepeat"), "repeat"),
            Example.create(Arrays.asList("aaaaa", "aaaaaa"), "aaaaa"),
            Example.create(Arrays.asList("aaaaaa", "aaaaa"), "aaaaa"));
    String name = "longestCommonSuffix";
    String description = "find the longest common suffix of two words";
    String expectedProgram =
        "RIGHT(var_0, ARRAYFORMULA(SUM(IF(EQ(RIGHT(var_0, SEQUENCE(MIN(LEN(var_0), LEN(var_1)))),"
            + " RIGHT(var_1, SEQUENCE(MIN(LEN(var_0), LEN(var_1))))), 1, 0))))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.ARRAY, BenchmarkTag.TOO_DIFFICULT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark prefixRemainder() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("prefix", "pre"), "fix"),
            Example.create(Arrays.asList("nonsense", "non"), "sense"),
            Example.create(Arrays.asList("atypical", "a"), "typical"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a", "a"), ""),
            Example.create(Arrays.asList("aaaaaa", "aa"), "aaaa"),
            Example.create(Arrays.asList("aaaaaa", "aaaaa"), "a"),
            Example.create(Arrays.asList("word", "word"), ""),
            Example.create(Arrays.asList("substrings", "substring"), "s"),
            Example.create(Arrays.asList("repeatedrepeat", "repeat"), "edrepeat"),
            Example.create(Arrays.asList("repeatrepeat", "repeat"), "repeat"),
            Example.create(Arrays.asList("abcdefg", "abcd"), "efg"));
    String name = "prefixRemainder";
    String description = "extract the rest of a word given a prefix";
    String expectedProgram = "RIGHT(var_0, MINUS(LEN(var_0), LEN(var_1)))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of();
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark suffixRemainder() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("suffix", "fix"), "suf"),
            Example.create(Arrays.asList("declaration", "tion"), "declara"),
            Example.create(Arrays.asList("analyticly", "ly"), "analytic"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a", "a"), ""),
            Example.create(Arrays.asList("aaaaaa", "aa"), "aaaa"),
            Example.create(Arrays.asList("aaaaaa", "aaaaa"), "a"),
            Example.create(Arrays.asList("word", "word"), ""),
            Example.create(Arrays.asList("asubstring", "substring"), "a"),
            Example.create(Arrays.asList("repeatedrepeat", "repeat"), "repeated"),
            Example.create(Arrays.asList("repeatrepeat", "repeat"), "repeat"),
            Example.create(Arrays.asList("abcdefg", "defg"), "abc"));
    String name = "suffixRemainder";
    String description = "extract the rest of a word given a suffix";
    String expectedProgram = "LEFT(var_0, MINUS(LEN(var_0), LEN(var_1)))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of();
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark stringEqual() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("string", "string"), "yes"),
            Example.create(Arrays.asList("string", "STRING"), "no"),
            Example.create(Arrays.asList("match this", "match this"), "yes"),
            Example.create(Arrays.asList("match this", "match that"), "no"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a", "a"), "yes"),
            Example.create(Arrays.asList("a", "b"), "no"),
            Example.create(Arrays.asList("A", "a"), "no"),
            Example.create(Arrays.asList(" ", " "), "yes"),
            Example.create(Arrays.asList("   ", "   "), "yes"),
            Example.create(Arrays.asList("   ", "    "), "no"),
            Example.create(Arrays.asList("some random text", "some random text"), "yes"),
            Example.create(Arrays.asList("some random-text", "some random text"), "no"));
    String name = "stringEqual";
    String description = "whether the two strings are exactly equal, yes or no";
    String expectedProgram = "IF(EXACT(var_0, var_1), \"yes\", \"no\")";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark stringEqualIgnoreCase() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("text", "text"), "yes"),
            Example.create(Arrays.asList("StRiNg", "sTrInG"), "yes"),
            Example.create(Arrays.asList("match this", "match this"), "yes"),
            Example.create(Arrays.asList("match those", "match that"), "no"),
            Example.create(Arrays.asList("substring", "substring."), "no"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a", "a"), "yes"),
            Example.create(Arrays.asList("a", "b"), "no"),
            Example.create(Arrays.asList("A", "a"), "yes"),
            Example.create(Arrays.asList(" ", " "), "yes"),
            Example.create(Arrays.asList("   ", "   "), "yes"),
            Example.create(Arrays.asList("   ", "    "), "no"),
            Example.create(Arrays.asList("some random text", "some random text"), "yes"),
            Example.create(Arrays.asList("some RaNdoM text", "some rAnDom text"), "yes"),
            Example.create(Arrays.asList("some RaNdoM text", "some rAnDom-text"), "no"));
    String name = "stringEqualIgnoreCase";
    String description = "whether the two strings are exactly equal ignoring case, yes or no";
    String expectedProgram = "IF(EXACT(LOWER(var_0), LOWER(var_1)), \"yes\", \"no\")";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark isLowerCase() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("this text is lowercase"), "true"),
            Example.create(Arrays.asList("Not Lowercase"), "false"),
            Example.create(Arrays.asList("123"), "true"),
            Example.create(Arrays.asList("XYZ"), "false"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("text"), "true"),
            Example.create(Arrays.asList("TEXT"), "false"),
            Example.create(Arrays.asList("tExt"), "false"),
            Example.create(Arrays.asList("1"), "true"),
            Example.create(Arrays.asList("1a"), "true"),
            Example.create(Arrays.asList("1A"), "false"),
            Example.create(Arrays.asList("a"), "true"),
            Example.create(Arrays.asList("A"), "false"));
    String name = "isLowerCase";
    String description = "whether the string is lowercase";
    String expectedProgram = "IF(EXACT(var_0, LOWER(var_0)), \"true\", \"false\")";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark containsString() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("same", "same"), "TRUE"),
            Example.create(Arrays.asList("the text to search in", "find this"), "FALSE"),
            Example.create(Arrays.asList("the text to search in", "SEARCH"), "FALSE"),
            Example.create(Arrays.asList("the text to search in", "search"), "TRUE"),
            Example.create(Arrays.asList("the text to search in", "t to s"), "TRUE"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1234567890", "5678"), "TRUE"),
            Example.create(Arrays.asList("1234567890", "5768"), "FALSE"),
            Example.create(Arrays.asList("alphabet", "alphabet"), "TRUE"),
            Example.create(Arrays.asList("alphabet", "alphabets"), "FALSE"),
            Example.create(Arrays.asList("Alphabet", "alphAbet"), "FALSE"),
            Example.create(Arrays.asList("Abcabc", "abc"), "TRUE"),
            Example.create(Arrays.asList("Abcabc", "abca"), "FALSE"),
            Example.create(Arrays.asList("a", "a"), "TRUE"),
            Example.create(Arrays.asList("a", "A"), "FALSE"),
            Example.create(Arrays.asList("a", "b"), "FALSE"));
    String name = "containsString";
    String description = "whether the first string contains the second";
    String expectedProgram = "TO_TEXT(ISNUMBER(FIND(var_1, var_0)))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark containsStringIgnoreCase() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("the text to search in", "find this"), "FALSE"),
            Example.create(Arrays.asList("the text to search in", "s.arch"), "FALSE"),
            Example.create(Arrays.asList("the text to search in", "SeArCh"), "TRUE"),
            Example.create(Arrays.asList("THE TEXT TO SEARCH IN", "t to s"), "TRUE"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1234567890", "5678"), "TRUE"),
            Example.create(Arrays.asList("1234567890", "5768"), "FALSE"),
            Example.create(Arrays.asList("alphabet", "alphabet"), "TRUE"),
            Example.create(Arrays.asList("alphabet", "alphabets"), "FALSE"),
            Example.create(Arrays.asList("Alphabet", "alphAbet"), "TRUE"),
            Example.create(Arrays.asList("Abcabc", "abc"), "TRUE"),
            Example.create(Arrays.asList("Abcabc", "abca"), "TRUE"),
            Example.create(Arrays.asList("a", "a"), "TRUE"),
            Example.create(Arrays.asList("a", "A"), "TRUE"),
            Example.create(Arrays.asList("a", "b"), "FALSE"));
    String name = "containsStringIgnoreCase";
    String description = "whether the first string contains the second, ignoring case";
    String expectedProgram = "TO_TEXT(ISNUMBER(FIND(LOWER(var_1), LOWER(var_0))))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark secondStringIfFirstIsNone() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("primary", "secondary"), "primary"),
            Example.create(Arrays.asList("ABC", "XYZ"), "ABC"),
            Example.create(Arrays.asList("NOT NONE", "123"), "NOT NONE"),
            Example.create(Arrays.asList("NONE", "select this instead"), "select this instead"),
            Example.create(Arrays.asList("NONE", "<BACKUP>"), "<BACKUP>"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1", "2"), "1"),
            Example.create(Arrays.asList("2", "1"), "2"),
            Example.create(Arrays.asList("NONE", "NONE"), "NONE"),
            Example.create(Arrays.asList("none", "NONE"), "none"),
            Example.create(Arrays.asList("NONE", "??"), "??"),
            Example.create(Arrays.asList("a", "b"), "a"));
    String name = "secondStringIfFirstIsNone";
    String description = "select the first string, or the second if the first is NONE";
    String expectedProgram = "IF(EXACT(var_0, \"NONE\"), var_1, var_0)";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark selectLongerString() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("length,", "is same"), "length,"),
            Example.create(Arrays.asList("123", "45"), "123"),
            Example.create(Arrays.asList("xyz", "abcdef"), "abcdef"),
            Example.create(Arrays.asList("aa", "aaa"), "aaa"),
            Example.create(Arrays.asList("aa", "a"), "aa"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1234567890", "1234567890!"), "1234567890!"),
            Example.create(Arrays.asList("1234567890!", "1234567890"), "1234567890!"),
            Example.create(Arrays.asList("identical", "identical"), "identical"),
            Example.create(Arrays.asList("   ", "  "), "   "),
            Example.create(Arrays.asList("  ", "   "), "   "),
            Example.create(Arrays.asList("a", "this is a long string"), "this is a long string"),
            Example.create(Arrays.asList("also a long string", "!"), "also a long string"));
    String name = "selectLongerString";
    String description = "select the longer of 2 strings, defaulting to the first if equal length";
    String expectedProgram = "IF(GT(LEN(var_1), LEN(var_0)), var_1, var_0)";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark swapCase() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("lowercase"), "LOWERCASE"),
            Example.create(Arrays.asList("UPPERCASE"), "uppercase"),
            Example.create(Arrays.asList("swap"), "SWAP"),
            Example.create(Arrays.asList("CASE"), "case"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a"), "A"),
            Example.create(Arrays.asList("A"), "a"),
            Example.create(Arrays.asList("123"), "123"),
            Example.create(Arrays.asList(" "), " "),
            Example.create(Arrays.asList("12345abcd"), "12345ABCD"),
            Example.create(Arrays.asList("12345ABCD"), "12345abcd"));
    String name = "swapCase";
    String description = "swap the case of a string that is entirely uppercase or lowercase";
    String expectedProgram = "IF(EXACT(var_0, LOWER(var_0)), UPPER(var_0), LOWER(var_0))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark countStringOccurrences() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("the text to search in", "to"), "1"),
            Example.create(Arrays.asList("the text to search in", " "), "4"),
            Example.create(Arrays.asList("the text to search in", "t"), "4"),
            Example.create(Arrays.asList("the text to search in", " t"), "2"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("aaaa", "a"), "4"),
            Example.create(Arrays.asList("aaaa", "aa"), "2"),
            Example.create(Arrays.asList("aaaa", "aaa"), "1"),
            Example.create(Arrays.asList("aaaa", "aaaa"), "1"),
            Example.create(Arrays.asList("aaaa", "aaaaa"), "0"),
            Example.create(Arrays.asList("aaaa", "A"), "0"),
            Example.create(Arrays.asList("insinsideide", "inside"), "1"),
            Example.create(Arrays.asList("repeatrepeatrepeat", "repeat"), "3"),
            Example.create(Arrays.asList("repeatrePeatrepeat", "repeat"), "2"));
    String name = "countStringOccurrences";
    String description = "count the number of times the second string appears in the first";
    String expectedProgram =
        "TO_TEXT(DIVIDE(MINUS(LEN(var_0), LEN(SUBSTITUTE(var_0, var_1, \"\"))), LEN(var_1)))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of();
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark extractUrl1() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("https://www.google.com/"), "www.google.com"),
            Example.create(Arrays.asList("http://www.stanford.edu/news/"), "www.stanford.edu"),
            Example.create(Arrays.asList("http://a.b.c.org/d/e/f.g"), "a.b.c.org"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("https://google.com/"), "google.com"),
            Example.create(Arrays.asList("https://google.com/stuff"), "google.com"),
            Example.create(Arrays.asList("http://www.stanford.edu/"), "www.stanford.edu"),
            Example.create(Arrays.asList("http://a.b.c.org/"), "a.b.c.org"),
            Example.create(Arrays.asList("http://a.b/c.org/"), "a.b"),
            Example.create(Arrays.asList("https://a.b/c.org"), "a.b"),
            Example.create(Arrays.asList("http://a.b/x.y.z"), "a.b"),
            Example.create(Arrays.asList("http://1.23.4/5/6.7/"), "1.23.4"));
    String name = "extractUrl1";
    String description = "extract the part of a URL between the 2nd and 3rd slash";
    String expectedProgram =
        "MID(var_0, ADD(FIND(\"//\", var_0), 2), MINUS(MINUS(FIND(\"/\", var_0, 9), FIND(\"/\","
            + " var_0)), 2))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark extractUrl2() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("https://www.google.com/"), "/"),
            Example.create(Arrays.asList("http://www.stanford.edu/news/"), "/news/"),
            Example.create(Arrays.asList("http://a.b.c.org/d/e/f.g"), "/d/e/f.g"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("https://google.com/"), "/"),
            Example.create(Arrays.asList("https://google.com/stuff"), "/stuff"),
            Example.create(Arrays.asList("https://google.com/stuff/"), "/stuff/"),
            Example.create(Arrays.asList("http://www.stanford.edu/"), "/"),
            Example.create(Arrays.asList("http://a.b.c.org/"), "/"),
            Example.create(Arrays.asList("http://a.b/c.org/"), "/c.org/"),
            Example.create(Arrays.asList("https://a.b/c.org"), "/c.org"),
            Example.create(Arrays.asList("http://a.b/x.y.z"), "/x.y.z"),
            Example.create(Arrays.asList("http://1.23.4/5/6.7/"), "/5/6.7/"));
    String name = "extractUrl2";
    String description = "extract the part of a URL starting from the 3rd slash";
    String expectedProgram =
        "RIGHT(var_0, ADD(1, MINUS(LEN(var_0), FIND(\"/\", var_0, ADD(FIND(\"//\", var_0), 2)))))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark extractUrl3() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("https://www.google.com/"), "com"),
            Example.create(Arrays.asList("http://www.stanford.edu/news/"), "edu"),
            Example.create(Arrays.asList("http://a.b.c.de/f/g.h"), "de"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("https://google.com/"), "com"),
            Example.create(Arrays.asList("https://google.com/stuff"), "com"),
            Example.create(Arrays.asList("http://www.stanford.edu/"), "edu"),
            Example.create(Arrays.asList("http://a.b.c.org/"), "org"),
            Example.create(Arrays.asList("http://a.b/c.org/"), "b"),
            Example.create(Arrays.asList("https://a.b/c.org"), "b"),
            Example.create(Arrays.asList("http://a.bcd/x.y.z"), "bcd"),
            Example.create(Arrays.asList("http://1.23.4/5/6.7/"), "4"));
    String name = "extractUrl3";
    String description = "extract the top-level domain of a URL";
    String expectedProgram = "REGEXEXTRACT(var_0, \"//[^/]*\\.(.+?)/\")";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.REGEX);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark concatSorted() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("orange", "apple", "banana"), "apple banana orange"),
            Example.create(Arrays.asList("cat", "bird", "dog"), "bird cat dog"),
            Example.create(Arrays.asList("red", "yellow", "blue"), "blue red yellow"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("same", "same", "same"), "same same same"),
            Example.create(Arrays.asList("same", "different", "same"), "different same same"),
            Example.create(Arrays.asList("apple", "banana", "orange"), "apple banana orange"),
            Example.create(Arrays.asList("apple", "orange", "banana"), "apple banana orange"),
            Example.create(Arrays.asList("banana", "apple", "orange"), "apple banana orange"),
            Example.create(Arrays.asList("banana", "orange", "apple"), "apple banana orange"),
            Example.create(Arrays.asList("orange", "banana", "apple"), "apple banana orange"),
            Example.create(Arrays.asList("rad", "radish", "radius"), "rad radish radius"),
            Example.create(Arrays.asList("radius", "rad", "radish"), "rad radish radius"));
    String name = "concatSorted";
    String description = "concatenate 3 strings in alphabetical order";
    String expectedProgram = "JOIN(\" \", TRANSPOSE(SORT(TRANSPOSE({var_0, var_1, var_2}))))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.ARRAY);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark concatVariableNumber() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("orange", "apple", "banana"), "orange, apple, banana"),
            Example.create(Arrays.asList("cat", "bird", ""), "cat, bird"),
            Example.create(Arrays.asList("red", "", ""), "red"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a", "b", "c"), "a, b, c"),
            Example.create(Arrays.asList("a", "b", ""), "a, b"),
            Example.create(Arrays.asList("a", "", ""), "a"),
            Example.create(
                Arrays.asList("multiple words", "with, 'punctuation'", "123"),
                "multiple words, with, 'punctuation', 123"),
            Example.create(
                Arrays.asList("multiple words", "with, 'punctuation'", ""),
                "multiple words, with, 'punctuation'"),
            Example.create(Arrays.asList("with, comma", "", ""), "with, comma"),
            Example.create(Arrays.asList("AAA", "AA", "A"), "AAA, AA, A"),
            Example.create(Arrays.asList("AAAA", "AA", ""), "AAAA, AA"),
            Example.create(Arrays.asList("AAAAAA", "", ""), "AAAAAA"));
    String name = "concatVariableNumber";
    String description = "concatenate a variable number of strings";
    String expectedProgram =
        "IFNA(JOIN(\", \", FILTER(TRANSPOSE({var_0, var_1, var_2}), NOT(EQ(0,"
            + " LEN(TRANSPOSE({var_0, var_1, var_2})))))), \"\")";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(
            BenchmarkTag.CONSTANT,
            BenchmarkTag.CONDITIONAL,
            BenchmarkTag.ARRAY,
            BenchmarkTag.TOO_DIFFICULT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark capitalizeCityAndState() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("mountain view, ca"), "Mountain View, CA"),
            Example.create(Arrays.asList("HOUSTON, TX"), "Houston, TX"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("seattle, wa"), "Seattle, WA"),
            Example.create(Arrays.asList("St. LOUIS, mo"), "St. Louis, MO"),
            Example.create(Arrays.asList("sAcRaMeNtO, Ca"), "Sacramento, CA"),
            Example.create(Arrays.asList("new York, nY"), "New York, NY"),
            Example.create(Arrays.asList("a, bc"), "A, BC"),
            Example.create(Arrays.asList("aa aaa a, bc"), "Aa Aaa A, BC"));
    String name = "capitalizeCityAndState";
    String description = "fix capitalization of city and state";
    String expectedProgram =
        "CONCATENATE(LEFT(PROPER(var_0), MINUS(LEN(var_0), 1)), UPPER(RIGHT(var_0, 1)))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of();
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark capitalizeSentence() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("this is a sentence."), "This is a sentence."),
            Example.create(Arrays.asList("FIX CAPITALIZATION."), "Fix capitalization."));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a!"), "A!"),
            Example.create(Arrays.asList("A!"), "A!"),
            Example.create(Arrays.asList("a a?"), "A a?"),
            Example.create(Arrays.asList("a A?"), "A a?"),
            Example.create(Arrays.asList("abC DEFGH iJ"), "Abc defgh ij"),
            Example.create(Arrays.asList("aaaaaa"), "Aaaaaa"),
            Example.create(Arrays.asList("12345 XYZ"), "12345 xyz"));
    String name = "capitalizeSentence";
    String description = "capitalize the first word and lowercase the rest";
    String expectedProgram = "REPLACE(LOWER(var_0), 1, 1, UPPER(LEFT(var_0, 1)))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of();
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark abbreviateDate() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("Wednesday, April 8"), "WED, APR 8"),
            Example.create(Arrays.asList("Friday, February 21"), "FRI, FEB 21"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("Sunday, January 30"), "SUN, JAN 30"),
            Example.create(Arrays.asList("Monday, March 3"), "MON, MAR 3"),
            Example.create(Arrays.asList("Tuesday, May 15"), "TUE, MAY 15"),
            Example.create(Arrays.asList("Thursday, June 7"), "THU, JUN 7"),
            Example.create(Arrays.asList("Saturday, July 4"), "SAT, JUL 4"),
            Example.create(Arrays.asList("Wednesday, September 28"), "WED, SEP 28"));
    String name = "abbreviateDate";
    String description = "abbreviate day of week and month names with capital letters";
    String expectedProgram =
        "CONCATENATE(LEFT(UPPER(var_0), 3), \", \", MID(UPPER(var_0), ADD(FIND(\",\", var_0), 2),"
            + " 3), MID(var_0, FIND(\" \", var_0, ADD(FIND(\",\", var_0), 2)), 3))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.TOO_DIFFICULT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark twoLetterAcronym() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("Product Area"), "PA"),
            Example.create(Arrays.asList("Vice President"), "VP"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("Principal Investigator"), "PI"),
            Example.create(Arrays.asList("Medical Doctor"), "MD"),
            Example.create(Arrays.asList("Artificial Intelligence"), "AI"),
            Example.create(Arrays.asList("Reinforcement Learning"), "RL"),
            Example.create(Arrays.asList("Programming Languages"), "PL"),
            Example.create(Arrays.asList("Abc Xyz"), "AX"));
    String name = "twoLetterAcronym";
    String description = "create acronym from two words in one cell";
    String expectedProgram =
        "CONCATENATE(LEFT(var_0, 1), MID(var_0, ADD(FIND(\" \", var_0), 1), 1))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark twoLetterAcronymCapitalization() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("product area"), "PA"),
            Example.create(Arrays.asList("Vice president"), "VP"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("principal investigator"), "PI"),
            Example.create(Arrays.asList("medical Doctor"), "MD"),
            Example.create(Arrays.asList("ARTIFICIAL INTELLIGENCE"), "AI"),
            Example.create(Arrays.asList("reinforcement learning"), "RL"),
            Example.create(Arrays.asList("pROGRAMMING lANGUAGES"), "PL"),
            Example.create(Arrays.asList("abc xyz"), "AX"));
    String name = "twoLetterAcronymCapitalization";
    String description = "create capitalized acronym from two words in one cell";
    String expectedProgram =
        "UPPER(CONCATENATE(LEFT(var_0, 1), MID(var_0, ADD(FIND(\" \", var_0), 1), 1)))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark multiLetterAcronym() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("Chief Executive Officer"), "CEO"),
            Example.create(Arrays.asList("Vice President"), "VP"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("Oneword"), "O"),
            Example.create(Arrays.asList("Reinforcement Learning"), "RL"),
            Example.create(Arrays.asList("Central Intelligence Agency"), "CIA"),
            Example.create(Arrays.asList("Principles Of Programming Languages"), "POPL"),
            Example.create(Arrays.asList("Self Contained Underwater Breathing Apparatus"), "SCUBA"),
            Example.create(Arrays.asList("Better Business Bureau"), "BBB"));
    String name = "multiLetterAcronym";
    String description = "create acronym from multiple words in one cell";
    String expectedProgram = "JOIN(\"\", ARRAYFORMULA(LEFT(SPLIT(var_0, \" \"), 1)))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.ARRAY);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark multiLetterAcronymCapitalization() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("chief executive officer"), "CEO"),
            Example.create(Arrays.asList("Vice president"), "VP"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("oneword"), "O"),
            Example.create(Arrays.asList("reinforcement Learning"), "RL"),
            Example.create(Arrays.asList("central intelligence agency"), "CIA"),
            Example.create(Arrays.asList("Principles of Programming Languages"), "POPL"),
            Example.create(Arrays.asList("self contained underwater breathing apparatus"), "SCUBA"),
            Example.create(Arrays.asList("BETTER BUSINESS BUREAU"), "BBB"));
    String name = "multiLetterAcronymCapitalization";
    String description = "create capitalized acronym from multiple words in one cell";
    String expectedProgram = "UPPER(JOIN(\"\", ARRAYFORMULA(LEFT(SPLIT(var_0, \" \"), 1))))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.ARRAY);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark reduceAny() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("no", "no", "no"), "no"),
            Example.create(Arrays.asList("no", "yes", "no"), "yes"),
            Example.create(Arrays.asList("no", "maybe", "no"), "no"),
            Example.create(Arrays.asList("maybe", "maybe", "maybe"), "no"),
            Example.create(Arrays.asList("no", "no", "yes"), "yes"),
            Example.create(Arrays.asList("yes", "no", "no"), "yes"),
            Example.create(Arrays.asList("yes", "yes", "yes"), "yes"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("yes", "yes", "no"), "yes"),
            Example.create(Arrays.asList("yes", "no", "yes"), "yes"),
            Example.create(Arrays.asList("no", "yes", "yes"), "yes"),
            Example.create(Arrays.asList("maybe", "maybe", "yes"), "yes"),
            Example.create(Arrays.asList("maybe", "yes", "no"), "yes"),
            Example.create(Arrays.asList("no", "maybe", "yes"), "yes"),
            Example.create(Arrays.asList("yes", "no", "maybe"), "yes"),
            Example.create(Arrays.asList("maybe", "maybe", "no"), "no"),
            Example.create(Arrays.asList("maybe", "no", "no"), "no"),
            Example.create(Arrays.asList("no", "no", "maybe"), "no"));
    String name = "reduceAny";
    String description = "yes if at least one input string is yes";
    // Without ARRAYFORMULA:
    // IF(OR(EXACT(\"yes\", var_0), EXACT(\"yes\", var_1), EXACT(\"yes\", var_2)), \"yes\", \"no\")
    String expectedProgram =
        "IF(OR(ARRAYFORMULA(EXACT(\"yes\", {var_0, var_1, var_2}))), \"yes\", \"no\")";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL, BenchmarkTag.ARRAY);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark removeDuplicates() {
    List<Example> examples =
        Arrays.asList(Example.create(Arrays.asList("3 1 4 1 5 9 2 6 5 3 5 8"), "3 1 4 5 9 2 6 8"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("0"), "0"),
            Example.create(Arrays.asList("77777"), "77777"),
            Example.create(Arrays.asList("77 77"), "77"),
            Example.create(Arrays.asList("1 22 333 4444 333 22 1"), "1 22 333 4444"),
            Example.create(Arrays.asList("1 22 333 4444 33 2 11"), "1 22 333 4444 33 2 11"),
            Example.create(Arrays.asList("34 56 12 56 23 23 78 34 18"), "34 56 12 23 78 18"),
            Example.create(Arrays.asList("3 3 3 3 3 3 3 3 3 3 3 3"), "3"),
            Example.create(Arrays.asList("3 3 33 3 3 3 1 3 3 31 3 3"), "3 33 1 31"));
    String name = "removeDuplicates";
    String description = "remove duplicate numbers";
    String expectedProgram = "JOIN(\" \", UNIQUE(TRANSPOSE(SPLIT(var_0, \" \"))))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.ARRAY);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark stringLength() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("alphabet"), "8"),
            Example.create(Arrays.asList("google"), "6"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList(""), "0"),
            Example.create(Arrays.asList("7"), "1"),
            Example.create(Arrays.asList("This is a sentence."), "19"),
            Example.create(Arrays.asList("     "), "5"),
            Example.create(Arrays.asList("What's the length of this string?"), "33"));
    String name = "stringLength";
    String description = "length of string";
    String expectedProgram = "TO_TEXT(LEN(var_0))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of();
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark pathDepth() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("/this/is/a/path"), "4"),
            Example.create(Arrays.asList("/home"), "1"),
            Example.create(Arrays.asList("/a/b"), "2"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("/this/is/a/very/long/path"), "6"),
            Example.create(Arrays.asList("/x"), "1"),
            Example.create(Arrays.asList("/abcde"), "1"),
            Example.create(Arrays.asList("/a/c/e"), "3"),
            Example.create(Arrays.asList("/a/cde"), "2"),
            Example.create(Arrays.asList("/abc/e"), "2"));
    String name = "pathDepth";
    String description = "the depth of a path, i.e., count the number of /";
    String expectedProgram = "TO_TEXT(MINUS(LEN(var_0), LEN(SUBSTITUTE(var_0, \"/\", \"\"))))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark numberOrWord() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("3.14"), "number"),
            Example.create(Arrays.asList("314159"), "number"),
            Example.create(Arrays.asList("text"), "word"),
            Example.create(Arrays.asList("Google"), "word"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("0.000001"), "number"),
            Example.create(Arrays.asList("1234567890123456789"), "number"),
            Example.create(Arrays.asList("99999"), "number"),
            Example.create(Arrays.asList("7"), "number"),
            Example.create(Arrays.asList("9.9"), "number"),
            Example.create(Arrays.asList("a"), "word"),
            Example.create(Arrays.asList("A"), "word"),
            Example.create(Arrays.asList("word"), "word"),
            Example.create(Arrays.asList("number"), "word"),
            Example.create(Arrays.asList("GraphSAGE"), "word"));
    String name = "numberOrWord";
    String description = "determine if the text is a word or a number";
    String expectedProgram = "IF(REGEXMATCH(var_0, \"^[[:alpha:]]+$\"), \"word\", \"number\")";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL, BenchmarkTag.REGEX);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark numberOrWordOrNeither() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("3.14"), "number"),
            Example.create(Arrays.asList("314159"), "number"),
            Example.create(Arrays.asList("text"), "word"),
            Example.create(Arrays.asList("Google"), "word"),
            Example.create(Arrays.asList("python3"), "neither"),
            Example.create(Arrays.asList("google.com"), "neither"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("12345.67890123456789"), "number"),
            Example.create(Arrays.asList("99999"), "number"),
            Example.create(Arrays.asList("7"), "number"),
            Example.create(Arrays.asList("I"), "word"),
            Example.create(Arrays.asList("word"), "word"),
            Example.create(Arrays.asList("number"), "word"),
            Example.create(Arrays.asList("area120"), "neither"),
            Example.create(Arrays.asList(" "), "neither"),
            Example.create(Arrays.asList("two words"), "neither"),
            Example.create(Arrays.asList("1.2.3"), "neither"));
    String name = "numberOrWordOrNeither";
    String description = "determine if the text is a word, number, or neither";
    String expectedProgram =
        "IF(REGEXMATCH(var_0, \"^[[:alpha:]]+$\"), \"word\", IF(REGEXMATCH(var_0,"
            + " \"^\\d+(\\.\\d+)?$\"), \"number\", \"neither\"))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL, BenchmarkTag.REGEX);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark truncateIfTooLong() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("short text"), "short text"),
            Example.create(Arrays.asList("max length text"), "max length text"),
            Example.create(Arrays.asList("extremely lengthy text"), "extremely lengt..."),
            Example.create(
                Arrays.asList("this text should be truncated"), "this text shoul...")); // NOTYPO
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a"), "a"),
            Example.create(Arrays.asList("  "), "  "),
            Example.create(Arrays.asList("1234567890123456789"), "123456789012345..."),
            Example.create(Arrays.asList("12345678901234567"), "123456789012345..."),
            Example.create(Arrays.asList("1234567890123456"), "123456789012345..."),
            Example.create(Arrays.asList("123456789012345"), "123456789012345"),
            Example.create(Arrays.asList("12345678901234"), "12345678901234"),
            Example.create(Arrays.asList("Acknowledgements"), "Acknowledgement..."),
            Example.create(Arrays.asList("Acknowledgement"), "Acknowledgement"),
            Example.create(Arrays.asList("infrastructure"), "infrastructure"));
    String name = "truncateIfTooLong";
    String description = "truncate and add ... if longer than 15 characters";
    String expectedProgram = "IF(GT(LEN(var_0), 15), CONCATENATE(LEFT(var_0, 15), \"...\"), var_0)";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark dateTransformation1() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("08092019"), "09/08/2019"),
            Example.create(Arrays.asList("12032020"), "03/12/2020"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("30092016"), "09/30/2016"),
            Example.create(Arrays.asList("23121637"), "12/23/1637"),
            Example.create(Arrays.asList("01023456"), "02/01/3456"),
            Example.create(Arrays.asList("11111111"), "11/11/1111"),
            Example.create(Arrays.asList("10101010"), "10/10/1010"));
    String name = "dateTransformation1";
    String description = "change DDMMYYYY date to MM/DD/YYYY";
    String expectedProgram = "CONCATENATE(MID(var_0, 3, 2), \"/\", REPLACE(var_0, 3, 2, \"/\"))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark dateTransformation2() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("2019-11-23"), "2019/11/23"),
            Example.create(Arrays.asList("2020-03-07"), "2020/03/07"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("2016-09-30"), "2016/09/30"),
            Example.create(Arrays.asList("1637-12-23"), "1637/12/23"),
            Example.create(Arrays.asList("3456-02-01"), "3456/02/01"),
            Example.create(Arrays.asList("1111-11-11"), "1111/11/11"),
            Example.create(Arrays.asList("1010-10-10"), "1010/10/10"));
    String name = "dateTransformation2";
    String description = "change YYYY-MM-DD date to YYYY/MM/DD";
    String expectedProgram = "SUBSTITUTE(var_0, \"-\", \"/\")";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark dateTransformation3() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("2019-11-23"), "11/23"),
            Example.create(Arrays.asList("2020-03-07"), "03/07"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("2016-09-30"), "09/30"),
            Example.create(Arrays.asList("1637-12-23"), "12/23"),
            Example.create(Arrays.asList("3456-02-01"), "02/01"),
            Example.create(Arrays.asList("1111-11-11"), "11/11"),
            Example.create(Arrays.asList("1010-10-10"), "10/10"));
    String name = "dateTransformation3";
    String description = "change YYYY-MM-DD date to MM/DD";
    String expectedProgram = "SUBSTITUTE(RIGHT(var_0, 5), \"-\", \"/\")";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark padSpaceToGivenWidth() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("string", "10"), "    string"),
            Example.create(Arrays.asList("text", "5"), " text"),
            Example.create(Arrays.asList("multiple words", "17"), "   multiple words"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("string", "6"), "string"),
            Example.create(Arrays.asList("string", "7"), " string"),
            Example.create(Arrays.asList("string", "8"), "  string"),
            Example.create(Arrays.asList("text", "4"), "text"),
            Example.create(Arrays.asList(" ", "3"), "   "),
            Example.create(Arrays.asList("", "3"), "   "),
            Example.create(Arrays.asList("", "1"), " "),
            Example.create(Arrays.asList("", "0"), ""),
            Example.create(Arrays.asList("  a b c d e  ", "20"), "         a b c d e  "));
    String name = "padSpaceToGivenWidth";
    String description = "pad text with spaces to a given width";
    String expectedProgram = "CONCATENATE(REPT(\" \", MINUS(VALUE(var_1), LEN(var_0))), var_0)";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark padZerosToFixedWidth() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("123"), "00123"),
            Example.create(Arrays.asList("45"), "00045"),
            Example.create(Arrays.asList("6780"), "06780"),
            Example.create(Arrays.asList("9"), "00009"),
            Example.create(Arrays.asList("54321"), "54321"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("0"), "00000"),
            Example.create(Arrays.asList("3"), "00003"),
            Example.create(Arrays.asList("33"), "00033"),
            Example.create(Arrays.asList("303"), "00303"),
            Example.create(Arrays.asList("4444"), "04444"),
            Example.create(Arrays.asList("6000"), "06000"),
            Example.create(Arrays.asList("7007"), "07007"),
            Example.create(Arrays.asList("55555"), "55555"),
            Example.create(Arrays.asList("56078"), "56078"),
            Example.create(Arrays.asList("60000"), "60000"));
    String name = "padZerosToFixedWidth";
    String description = "pad number with 0 to width 5";
    String expectedProgram = "CONCATENATE(REPT(\"0\", MINUS(5, LEN(var_0))), var_0)";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.CONSTANT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark removeZeros() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("00123"), "123"),
            Example.create(Arrays.asList("00045"), "45"),
            Example.create(Arrays.asList("06080"), "6080"),
            Example.create(Arrays.asList("99999"), "99999"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("00003"), "3"),
            Example.create(Arrays.asList("00040"), "40"),
            Example.create(Arrays.asList("00066"), "66"),
            Example.create(Arrays.asList("00100"), "100"),
            Example.create(Arrays.asList("00203"), "203"),
            Example.create(Arrays.asList("00555"), "555"),
            Example.create(Arrays.asList("06000"), "6000"),
            Example.create(Arrays.asList("05030"), "5030"),
            Example.create(Arrays.asList("90909"), "90909"),
            Example.create(Arrays.asList("10000"), "10000"));
    String name = "removeZeros";
    String description = "remove zeros at the beginning";
    String expectedProgram =
        "RIGHT(var_0, MINUS(5, SUM(ARRAYFORMULA(IF(EXACT(LEFT(var_0, SEQUENCE(1, 5)), REPT(\"0\","
            + " SEQUENCE(1, 5))), 1, 0)))))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.ARRAY, BenchmarkTag.TOO_DIFFICULT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark removeWhitespaceLowercase() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("  TEXT "), "text"),
            Example.create(Arrays.asList("Two words    "), "two words"),
            Example.create(Arrays.asList("   before and after  "), "before and after"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("a"), "a"),
            Example.create(Arrays.asList("                   A"), "a"),
            Example.create(Arrays.asList("X                   "), "x"),
            Example.create(Arrays.asList("             a B                  "), "a b"),
            Example.create(Arrays.asList(" "), ""),
            Example.create(Arrays.asList("  . "), "."),
            Example.create(Arrays.asList("    "), ""),
            Example.create(Arrays.asList("nothing to change!"), "nothing to change!"),
            Example.create(Arrays.asList("Nothing to Remove!"), "nothing to remove!"),
            Example.create(
                Arrays.asList(" multiple SPACEs to remove... "), "multiple spaces to remove..."));
    String name = "removeWhitespaceLowercase";
    String description = "remove leading and trailing spaces and tabs, and lowercase";
    String expectedProgram = "TRIM(LOWER(var_0))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of();
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark addDecimal() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("1.23"), "1.23"),
            Example.create(Arrays.asList("45"), "45.0"),
            Example.create(Arrays.asList("67.0"), "67.0"),
            Example.create(Arrays.asList("9876"), "9876.0"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1.111"), "1.111"),
            Example.create(Arrays.asList("11111"), "11111.0"),
            Example.create(Arrays.asList("0.00000001"), "0.00000001"),
            Example.create(Arrays.asList("1000000000"), "1000000000.0"),
            Example.create(Arrays.asList("10000000.0"), "10000000.0"),
            Example.create(Arrays.asList("54321.54321"), "54321.54321"),
            Example.create(Arrays.asList("54321054321"), "54321054321.0"),
            Example.create(Arrays.asList("0"), "0.0"),
            Example.create(Arrays.asList("1"), "1.0"),
            Example.create(Arrays.asList("1.1"), "1.1"));
    String name = "addDecimal";
    String description = "add decimal point if not present";
    String expectedProgram = "IF(ISERROR(FIND(\".\", var_0)), CONCATENATE(var_0, \".0\"), var_0)";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark addPlusSign() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("12"), "+12"),
            Example.create(Arrays.asList("-34"), "-34"),
            Example.create(Arrays.asList("567"), "+567"),
            Example.create(Arrays.asList("-8"), "-8"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1"), "+1"),
            Example.create(Arrays.asList("55"), "+55"),
            Example.create(Arrays.asList("1111"), "+1111"),
            Example.create(Arrays.asList("402340234"), "+402340234"),
            Example.create(Arrays.asList("-1"), "-1"),
            Example.create(Arrays.asList("-70"), "-70"),
            Example.create(Arrays.asList("-1111"), "-1111"),
            Example.create(Arrays.asList("-98765"), "-98765"));
    String name = "addPlusSign";
    String description = "add plus sign to positive integers";
    String expectedProgram = "IF(EXACT(LEFT(var_0, 1), \"-\"), var_0, CONCATENATE(\"+\", var_0))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark negativeInParentheses() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("12"), "12"),
            Example.create(Arrays.asList("-34"), "(34)"),
            Example.create(Arrays.asList("567"), "567"),
            Example.create(Arrays.asList("-8"), "(8)"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1"), "1"),
            Example.create(Arrays.asList("55"), "55"),
            Example.create(Arrays.asList("1111"), "1111"),
            Example.create(Arrays.asList("402340234"), "402340234"),
            Example.create(Arrays.asList("-1"), "(1)"),
            Example.create(Arrays.asList("-70"), "(70)"),
            Example.create(Arrays.asList("-1111"), "(1111)"),
            Example.create(Arrays.asList("-98765"), "(98765)"));
    String name = "negativeInParentheses";
    String description = "enclose negative numbers in parentheses";
    String expectedProgram =
        "IF(EXACT(LEFT(var_0, 1), \"-\"), CONCATENATE(SUBSTITUTE(var_0, \"-\", \"(\"), \")\"),"
            + " var_0)";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark addThousandsSeparator() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("1234567"), "1,234,567"),
            Example.create(Arrays.asList("34856"), "34,856"),
            Example.create(Arrays.asList("987"), "987"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1"), "1"),
            Example.create(Arrays.asList("22"), "22"),
            Example.create(Arrays.asList("842"), "842"),
            Example.create(Arrays.asList("7639"), "7,639"),
            Example.create(Arrays.asList("88888"), "88,888"),
            Example.create(Arrays.asList("265395"), "265,395"),
            Example.create(Arrays.asList("5000000"), "5,000,000"),
            Example.create(Arrays.asList("37489735"), "37,489,735"),
            Example.create(Arrays.asList("969345346"), "969,345,346"),
            Example.create(Arrays.asList("1234567890"), "1,234,567,890"));
    String name = "addThousandsSeparator";
    String description = "add thousands separator to number";
    String expectedProgram = "TEXT(var_0, \"#,###\")";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.TEXT_FORMATTING);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark normalizeCurrency() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("12"), "$12.00"),
            Example.create(Arrays.asList("3"), "$3.00"),
            Example.create(Arrays.asList("4.5"), "$4.50"),
            Example.create(Arrays.asList("67.89"), "$67.89"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("1"), "$1.00"),
            Example.create(Arrays.asList("635"), "$635.00"),
            Example.create(Arrays.asList("99999999"), "$99999999.00"),
            Example.create(Arrays.asList("0.2"), "$0.20"),
            Example.create(Arrays.asList("0.23"), "$0.23"),
            Example.create(Arrays.asList("2.3"), "$2.30"),
            Example.create(Arrays.asList("5273.38"), "$5273.38"),
            Example.create(Arrays.asList("987643.5"), "$987643.50"),
            Example.create(Arrays.asList("88888888.88"), "$88888888.88"),
            Example.create(Arrays.asList("88888888.8"), "$88888888.80"));
    String name = "normalizeCurrency";
    String description = "create currency string";
    String expectedProgram = "CONCATENATE(\"$\", TEXT(var_0, \"0.00\"))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.TEXT_FORMATTING);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark extractNumberBetweenSecondParenthesis() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("Mark Henry (2 alpha): Xyz 10:4-18 (21) abba"), "21"),
            Example.create(Arrays.asList("James (5 test alpha): Ab 4:4-19 (10)"), "10"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("Rar H Chles (11 10 great): AAB 10:6-20 (22) beta"), "22"),
            Example.create(Arrays.asList("Jerry James (15 nice): SS 12:12-16 (46)"), "46"));
    String name = "extractNumberBetweenSecondParenthesis";
    String description = "extract the number between the second parenthesis";
    String expectedProgram = "REGEXEXTRACT(var_0, \".*\\(.*\\).*\\((\\d+)\\).*\")";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.REGEX);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark extractBetweenHTMLTags() {
    List<Example> examples =
        Arrays.asList(
            Example.create(
                Arrays.asList(
                    "<a href=\"/024232301i000001LKHJAlksh\" target=\"_alpha\">America (NV - NY) -"
                        + " Marlos1</a>"),
                "America (NV - NY) - Marlos1"),
            Example.create(
                Arrays.asList(
                    "<a href=\"/1332301i000001swsw\" target=\"_alpha\">Honda Austin (TX) -"
                        + " Henry</a>"),
                "Honda Austin (TX) - Henry"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(
                Arrays.asList(
                    "<a href=\"/434301i000001LKHJAlksh\" target=\"_alpha\">Audi San Jaso (CA) -"
                        + " Salem</a>"),
                "Audi San Jaso (CA) - Salem"),
            Example.create(
                Arrays.asList(
                    "<a href=\"/1332301i02325301swsw\" target=\"_alpha\">Benz Redmond (WA - MA) -"
                        + " Michael</a>"),
                "Benz Redmond (WA - MA) - Michael"));
    String name = "extractBetweenHTMLTags";
    String description = "extract the text between alpha> and <";
    String expectedProgram = "REGEXEXTRACT(var_0, \">(.*)<\")";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.REGEX);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark multipleIfStatus() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("43%"), "In Progress"),
            Example.create(Arrays.asList("0%"), "Not Yet Started"),
            Example.create(Arrays.asList("52%"), "In Progress"),
            Example.create(Arrays.asList("100%"), "Completed"),
            Example.create(Arrays.asList("47%"), "In Progress"),
            Example.create(Arrays.asList("100%"), "Completed"),
            Example.create(Arrays.asList("58%"), "In Progress"),
            Example.create(Arrays.asList("0%"), "Not Yet Started"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("34%"), "In Progress"),
            Example.create(Arrays.asList("44%"), "In Progress"),
            Example.create(Arrays.asList("56%"), "In Progress"),
            Example.create(Arrays.asList("0%"), "Not Yet Started"));
    String name = "multipleIfStatus";
    String description =
        "output \"Completed\" if 100%, \"Not Yet Started\" if 0%, and \"In Progress\" if between"
            + " 0% and 100%";
    String expectedProgram =
        "IF(var_0=\"100%\", \"Completed\", IF(var_0=\"0%\", \"Not Yet Started\", \"In Progress\"))";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark extractDateAndTime() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("14-18 October 2018, 2pm to 4pm"), "14 October, 2pm"),
            Example.create(Arrays.asList("12 September 2018, 4pm to 5pm"), "12 September, 4pm"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("11-14 November 2018, 1pm to 2pm"), "11 November, 1pm"),
            Example.create(Arrays.asList("11 September 2018, 1pm to 5pm"), "11 September, 1pm"));
    String name = "extractDateAndTime";
    String description = "extract date and time from two different input formats.";
    String expectedProgram =
        "CONCATENATE(REGEXEXTRACT(var_0, \"\\d+\"), REGEXEXTRACT(var_0, \" \\w+\"),"
            + " REGEXEXTRACT(var_0, \", \\d+pm\"))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.REGEX);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark concatenateUnderscoreWithEmptyColumns() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("alpha", "beta", "gamma delta"), "alpha_beta_gamma delta"),
            Example.create(Arrays.asList("", "", "Mark"), "Mark"),
            Example.create(Arrays.asList("alpha", "abd", ""), "alpha_abd"),
            Example.create(Arrays.asList("", "beta delta", "delta"), "beta delta_delta"),
            Example.create(
                Arrays.asList("apples oranges", "", "bananas"), "apples oranges_bananas"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("fire", "water", "soil"), "fire_water_soil"),
            Example.create(Arrays.asList("alpha", "beta delta", ""), "alpha_beta delta"),
            Example.create(Arrays.asList("", "apples oranges", ""), "apples oranges"),
            Example.create(Arrays.asList("", "", ""), ""),
            Example.create(Arrays.asList("a", "", ""), "a"),
            Example.create(Arrays.asList("", "a", ""), "a"),
            Example.create(Arrays.asList("", "", "a"), "a"),
            Example.create(Arrays.asList("a", "b", ""), "a_b"),
            Example.create(Arrays.asList("a", "", "b"), "a_b"),
            Example.create(Arrays.asList("", "a", "b"), "a_b"));
    String name = "concatenateUnderscoreWithEmptyColumns";
    String description = "concatenate column values with underscore except when value is empty.";
    String expectedProgram =
        "CONCATENATE(var_0, IF(OR(var_0=\"\", AND(var_1=\"\", var_2=\"\")), \"\", \"_\"), var_1,"
            + " IF(OR(var_1=\"\", var_2=\"\"), \"\", \"_\"), var_2)";
    ImmutableList<BenchmarkTag> tags =
        ImmutableList.of(
            BenchmarkTag.CONSTANT, BenchmarkTag.CONDITIONAL, BenchmarkTag.TOO_DIFFICULT);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark extractNumberStartingWith20() {
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("alpha 7TH MAY  200024241"), "200024241"),
            Example.create(Arrays.asList("FAST SHIPPING DELIVERY 200024227"), "200024227"),
            Example.create(Arrays.asList("Beta 203024202 Alpha 520042234"), "203024202"),
            Example.create(Arrays.asList("Beta 203024202"), "203024202"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("KING ROG ALPHA 9TH JUNE: ORDER 200324204"), "200324204"),
            Example.create(Arrays.asList("42324 SHIPPING DELIVERY 200024127"), "200024127"),
            Example.create(Arrays.asList("Gamma Beta 203024202 Alpha 520042234"), "203024202"));
    String name = "extractNumberStartingWith20";
    String description = "extract the number starting with 20 in the input string.";
    String expectedProgram = "TRIM(REGEXEXTRACT(var_0, \"\\s20\\d+\"))";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.REGEX);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark zzPrimalityTestingShouldFail() {
    // Benchmark we expect to always fail. The name starts with "zz" to ensure this is at the end of
    // ALL_BENCHMARKS, which is sorted by benchmark name.
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("five"), "prime"),
            Example.create(Arrays.asList("six"), "composite"),
            Example.create(Arrays.asList("seven"), "prime"),
            Example.create(Arrays.asList("nine"), "composite"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("two"), "prime"),
            Example.create(Arrays.asList("three"), "prime"),
            Example.create(Arrays.asList("four"), "composite"),
            Example.create(Arrays.asList("eight"), "composite"),
            Example.create(Arrays.asList("ten"), "composite"),
            Example.create(Arrays.asList("eleven"), "prime"),
            Example.create(Arrays.asList("twelve"), "composite"),
            Example.create(Arrays.asList("thirteen"), "prime"),
            Example.create(Arrays.asList("fourteen"), "composite"),
            Example.create(Arrays.asList("seventeen"), "prime"));
    String name = "zzPrimalityTestingShouldFail";
    String description = "check if number represented by input string is prime";
    String expectedProgram = "";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.SHOULD_FAIL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }

  public static Benchmark zzSameInputDifferentOutputShouldFail() {
    // Benchmark we expect to always fail. The name starts with "zz" to ensure this is at the end of
    // ALL_BENCHMARKS, which is sorted by benchmark name.
    List<Example> examples =
        Arrays.asList(
            Example.create(Arrays.asList("input"), "output1"),
            Example.create(Arrays.asList("input"), "output2"));
    List<Example> testExamples =
        Arrays.asList(
            Example.create(Arrays.asList("input2"), "output2A"),
            Example.create(Arrays.asList("input2"), "output2B"));
    String name = "zzSameInputDifferentOutputShouldFail";
    String description = "produce two different outputs for the same input";
    String expectedProgram = "";
    ImmutableList<BenchmarkTag> tags = ImmutableList.of(BenchmarkTag.SHOULD_FAIL);
    return new Benchmark(name, description, examples, testExamples, expectedProgram, tags);
  }
}
