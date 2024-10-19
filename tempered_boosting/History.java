//

public class History {
  public static String[][] HISTORY = {
    {
      "1.0",
      "Tempered Boosting (NeurIPS'23) + Poincar\'e disk embedding for log-/logistic-boosted sets of"
          + " DTs"
    }
  };

  public static String CURRENT_HISTORY() {
    return "V" + HISTORY[HISTORY.length - 1][0] + " : " + HISTORY[HISTORY.length - 1][1] + ".";
  }
}
