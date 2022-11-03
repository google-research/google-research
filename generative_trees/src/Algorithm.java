// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

class Algorithm implements Debuggable {
  // General variables
  public static int MAX_DEPTH_DT = 10000;
  public static int MAX_DEPTH_GT = 10000;

  public static int FEAT_SIZE = 0;
  public static int FEAT_SIZE_2 = 0;

  public static String IRRELEVANT_STRING = "IRRELEVANT";
  public static int IRRELEVANT_INT = -1;
  public static boolean IRRELEVANT_BOOLEAN = false;
  public static int IRRELEVANT_MAX_INT = 10001;

  public static String STRATEGY_DT_GROW_ONE_LEAF_HEAVIEST = "STRATEGY_DT_GROW_ONE_LEAF_HEAVIEST";
  // STRATEGY_DT_GROW_ONE_LEAF_HEAVIEST: DT | leaf | picks the heaviest leaf wrt current examples

  public static String[] ALL_STRATEGIES_DT_GROW = {STRATEGY_DT_GROW_ONE_LEAF_HEAVIEST};

  public static int STRATEGY_DT_GROW(String s) {
    int i = 0;
    do {
      if (ALL_STRATEGIES_DT_GROW[i].equals(s)) return i;
      i++;
    } while (i < ALL_STRATEGIES_DT_GROW.length);
    return -1;
  }

  public static void CHECK_STRATEGY_DT_GROW_CONTAINS(String s) {
    if (STRATEGY_DT_GROW(s) == -1)
      Dataset.perror("Algorithm.class :: no such STRATEGY_DT_GROW as " + s);
  }

  public static String NO_DT_SPLIT_MAX_SIZE = "NO_DT_SPLIT_MAX_SIZE",
      NO_DT_SPLIT_MAX_DEPTH = "NO_DT_SPLIT_MAX_DEPTH",
      NO_DT_SPLIT_NO_SPLITTABLE_LEAF_FOUND = "NO_DT_SPLIT_NO_SPLITTABLE_LEAF_FOUND",
      DT_SPLIT_OK = "DT_SPLIT_OK";

  public static String GT_SPLIT_OK = "GT_SPLIT_OK";

  public static String NOW;

  Vector<Boost> all_algorithms;
  Domain myDomain;

  double alpha;

  public static String[] MONTHS = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
  };

  public static Random R = new Random();

  Algorithm(Domain dom) {
    all_algorithms = new Vector<>();
    myDomain = dom;
  }

  public static double RANDOM_P_NOT_HALF() {
    double vv;
    do {
      vv = R.nextDouble();
    } while (vv == 0.5);
    return vv;
  }

  public static double RANDOM_P_NOT(double p) {
    double vv;
    do {
      vv = R.nextDouble();
    } while (vv == p);
    return vv;
  }

  public static void INIT() {
    Calendar cal = Calendar.getInstance();

    NOW =
        Algorithm.MONTHS[cal.get(Calendar.MONTH)]
            + "_"
            + cal.get(Calendar.DAY_OF_MONTH)
            + "th__"
            + cal.get(Calendar.HOUR_OF_DAY)
            + "h_"
            + cal.get(Calendar.MINUTE)
            + "m_"
            + cal.get(Calendar.SECOND)
            + "s";
  }

  public void addAlgorithm(Vector all_params) {
    String strategy_dt_grow_one_leaf;
    int gaming_iters;

    int i = 0;
    String n = (String) all_params.elementAt(i); // 0
    i++;
    double alpha = Double.parseDouble((String) all_params.elementAt(i)); // 1
    i++;
    String strategy_game = (String) all_params.elementAt(i); // 2
    i++;

    strategy_dt_grow_one_leaf = Algorithm.STRATEGY_DT_GROW_ONE_LEAF_HEAVIEST;
    gaming_iters = Integer.parseInt((String) all_params.elementAt(i)); // 3
    i++;

    Boost.COPYCAT_GENERATE_WITH_WHOLE_GT =
        !Boolean.parseBoolean((String) all_params.elementAt(i)); // 4
    i++;

    all_algorithms.addElement(
        new Boost(
            myDomain,
            n, // keep
            alpha, // keep
            strategy_game, // keep
            strategy_dt_grow_one_leaf, // keep
            gaming_iters)); // keep
  }

  public Generator_Tree simple_go() {
    return ((Boost) all_algorithms.elementAt(0)).simple_boost(0);
  }
}
