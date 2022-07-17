// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

public class History {
  public static String[][] HISTORY = {
    {
      "1.0",
      "Copycat training Generative Trees, companion code to the ICML'22 paper \"Generative Trees:"
          + " Adversarial and Copycat\", by R. Nock and M. Guillame-Bert"
    }
  };

  public static String CURRENT_HISTORY() {
    return "V" + HISTORY[HISTORY.length - 1][0] + " : " + HISTORY[HISTORY.length - 1][1] + ".";
  }
}
