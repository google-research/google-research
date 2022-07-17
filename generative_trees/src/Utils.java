// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Utils & Exceptions
 *****/

class Utils implements Debuggable {
  int domain_id;
  public static Random r = new Random();

  public static Vector<Vector<String>> ALL_NON_TRIVIAL_SUBSETS(Vector<String> v) {
    Vector<Vector<String>> ret = new Vector<>();
    Vector<String> copycat = new Vector<>();
    int i;
    for (i = 0; i < v.size(); i++) copycat.addElement(new String((String) v.elementAt(i)));

    MOVE_IN(ret, copycat);

    ret.removeElementAt(0);
    ret.removeElementAt(ret.size() - 1);

    return ret;
  }

  public static Vector<Vector<String>> ALL_NON_TRIVIAL_BOUNDED_SUBSETS(
      Vector<String> v, int max_size) {
    Vector<Vector<String>> ret = new Vector<>();
    Vector<String> copycat = new Vector<>();
    int i;
    for (i = 0; i < v.size(); i++) copycat.addElement(new String((String) v.elementAt(i)));

    MOVE_IN(ret, copycat, max_size);

    ret.removeElementAt(0);
    if (max_size <= v.size()) ret.removeElementAt(ret.size() - 1);

    return ret;
  }

  public static void MOVE_IN(Vector<Vector<String>> grow, Vector shrink, int max_size) {
    if ((shrink == null) || (shrink.size() == 0))
      Dataset.perror("Utils.class :: MOVE_IN impossible because empty list of values");

    if (grow == null) Dataset.perror("Utils.class :: MOVE_IN impossible because empty grow list");

    String s = (String) shrink.elementAt(0);
    Vector<String> v;
    Vector<String> vv;

    if (grow.size() == 0) {
      v = new Vector<>();
      grow.addElement(v);
    }

    int i, sinit = grow.size(), j;
    for (i = 0; i < sinit; i++) {
      vv = (Vector<String>) grow.elementAt(i);
      if (vv.size() < max_size) {
        v = new Vector<>();
        if (vv.size() > 0)
          for (j = 0; j < vv.size(); j++) v.addElement(new String((String) vv.elementAt(j)));
        v.addElement(new String(s));
        grow.addElement(v);
      }
    }
    shrink.removeElementAt(0);
    if (shrink.size() > 0) MOVE_IN(grow, shrink, max_size);
  }

  public static void MOVE_IN(Vector<Vector<String>> grow, Vector shrink) {
    if ((shrink == null) || (shrink.size() == 0))
      Dataset.perror("Utils.class :: MOVE_IN impossible because empty list of values");

    if (grow == null) Dataset.perror("Utils.class :: MOVE_IN impossible because empty grow list");

    String s = (String) shrink.elementAt(0);
    Vector<String> v;
    Vector<String> vv;

    if (grow.size() == 0) {
      v = new Vector<>();
      grow.addElement(v);
    }

    int i, sinit = grow.size(), j;
    for (i = 0; i < sinit; i++) {
      vv = (Vector<String>) grow.elementAt(i);
      v = new Vector<>();
      if (vv.size() > 0)
        for (j = 0; j < vv.size(); j++) v.addElement(new String((String) vv.elementAt(j)));
      v.addElement(new String(s));
      grow.addElement(v);
    }
    shrink.removeElementAt(0);
    if (shrink.size() > 0) MOVE_IN(grow, shrink);
  }
}

class NoLeafFoundException extends Exception {
  public NoLeafFoundException(String s) {
    super(s);
  }
}
