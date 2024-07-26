// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

public class History {
  public static String[][] HISTORY = {
    {
      "1.0",
      "ICML'24 sub code for paper #5193, do not share, do not distribute, delete after review"
          + " process"
    }
  };

    public static String CURRENT_HISTORY(){
	return "V" + HISTORY[HISTORY.length-1][0] + " : " + HISTORY[HISTORY.length-1][1] + ".";
    }
}
