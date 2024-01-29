import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Utils
 *****/

class Utils implements Debuggable {
  int domain_id;
  public static Random R = new Random();

  public static String NOW;

  public static Vector ALL_NON_TRIVIAL_SUBSETS(Vector v) {
    Vector ret = new Vector();
    Vector copycat = new Vector();
    int i;
    for (i = 0; i < v.size(); i++) copycat.addElement(new String((String) v.elementAt(i)));

    MOVE_IN(ret, copycat);

    ret.removeElementAt(0);
    ret.removeElementAt(ret.size() - 1);

    return ret;
  }

  public static void MOVE_IN(Vector grow, Vector shrink) {
    if ((shrink == null) || (shrink.size() == 0))
      Dataset.perror("Utils.class :: MOVE_IN impossible because empty list of values");

    if (grow == null) Dataset.perror("Utils.class :: MOVE_IN impossible because empty grow list");

    String s = (String) shrink.elementAt(0);
    Vector v, vv;

    if (grow.size() == 0) {
      v = new Vector();
      grow.addElement(v);
    }

    int i, sinit = grow.size(), j;
    for (i = 0; i < sinit; i++) {
      vv = (Vector) grow.elementAt(i);
      v = new Vector();
      if (vv.size() > 0)
        for (j = 0; j < vv.size(); j++) v.addElement(new String((String) vv.elementAt(j)));
      v.addElement(new String(s));
      grow.addElement(v);
    }
    shrink.removeElementAt(0);
    if (shrink.size() > 0) MOVE_IN(grow, shrink);
  }

  public static void INIT() {
    Calendar cal = Calendar.getInstance();

    R = new Random();
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

  public static double COMPUTE_P(double wp, double wn) {
    double val;

    if (wp + wn > 0.0) val = (wp / (wp + wn));
    else val = 0.5;

    if (val != 0.5) return val;

    double vv = RANDOM_P_NOT_HALF();

    if (vv < 0.5) val -= EPS2;
    else val += EPS2;
    return val;
  }

  public static double RANDOM_P_NOT_HALF() {
    double vv;
    do {
      vv = R.nextDouble();
    } while (vv == 0.5);
    return vv;
  }
}
