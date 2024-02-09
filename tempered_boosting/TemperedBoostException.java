import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class TemperedBoostException
 *****/

public class TemperedBoostException extends Exception {
  public static int NB_NUMERICAL_ISSUES_ABSENT_CLASS;
  public static int NB_NUMERICAL_ISSUES_INFINITE_LEAF_LABEL;
  public static int NB_NUMERICAL_ISSUES_INFINITE_MU;
  public static int NB_ZERO_WEIGHTS;
  public static int NB_INFINITE_WEIGHTS;

  public static double MIN_WEIGHT;

  public static String NUMERICAL_ISSUES_ABSENT_CLASS = "NUMERICAL_ISSUES_ABSENT_CLASS",
      NUMERICAL_ISSUES_INFINITE_LEAF_LABEL = "NUMERICAL_ISSUES_INFINITE_LEAF_LABEL",
      NUMERICAL_ISSUES_INFINITE_MU = "NUMERICAL_ISSUES_INFINITE_MU",
      ZERO_WEIGHTS = "ZERO_WEIGHTS",
      INFINITE_WEIGHTS = "INFINITE_WEIGHTS";

  public static String[] COUNTS_LABELS = {
    NUMERICAL_ISSUES_ABSENT_CLASS,
    NUMERICAL_ISSUES_INFINITE_LEAF_LABEL,
    NUMERICAL_ISSUES_INFINITE_MU,
    ZERO_WEIGHTS,
    INFINITE_WEIGHTS
  };

  public static int[] COUNTS = new int[COUNTS_LABELS.length];

  public static int COUNTS_INDEX(String s) {
    int i = 0;
    while ((i < COUNTS_LABELS.length) && (!COUNTS_LABELS[i].equals(s))) i++;
    if (i == COUNTS_LABELS.length)
      Dataset.perror("TemperedBoostException.class :: no such TemperedBoostException as " + s);
    return i;
  }

  public static String STATUS() {
    int i;
    String v = "Exceptions summary count: {";
    for (i = 0; i < COUNTS_LABELS.length; i++) {
      v += COUNTS[i];
      if (i < COUNTS_LABELS.length - 1) v += ", ";
    }
    v += "}";
    return v;
  }

  public static void RESET_COUNTS() {
    int i;
    for (i = 0; i < COUNTS_LABELS.length; i++) COUNTS[i] = 0;
    MIN_WEIGHT = -1.0;
  }

  public static void ADD(String which_one_increment) {
    COUNTS[COUNTS_INDEX(which_one_increment)]++;
  }

  public TemperedBoostException(String which_one_increment) {
    super(which_one_increment);
  }
}
