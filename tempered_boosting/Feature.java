import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Feature
 *****/

class Feature implements Debuggable {

  public static String NOMINAL = "NOMINAL", CONTINUOUS = "CONTINUOUS", CLASS = "CLASS";

  public static String TYPE[] = {Feature.NOMINAL, Feature.CONTINUOUS, Feature.CLASS};
  public static boolean MODALITIES[] = {true, false, false};
  public static int CLASS_INDEX = 2;

  public static double MODALITY_PRESENT = 1.0, MODALITY_ABSENT = 0.0;
  public static double MINV = -1.0, MAXV = 1.0; // values used for normalization in [MINV, MAXV]
  public static double MAX_CLASS_MAGNITUDE = 1.0;

  String name;
  String type;

  double minn, maxx;

  Vector modalities; // applies to nominal feature

  Vector tests;

  // set of test values as returned by TEST_LIST

  public static Vector TEST_LIST(Feature f) {
    // generates a vector of a list of tests -- does not depend on training data for privacy
    // purposes
    // if continuous, list of evenly spaces ties
    // if nominal, list of partial non-empty subsets of the whole set

    Vector v = new Vector();
    if (IS_CONTINUOUS(f.type)) {
      v = null;
    } else if (HAS_MODALITIES(f.type)) {
      v = Utils.ALL_NON_TRIVIAL_SUBSETS(f.modalities);
    }

    return v;
  }

  public void update_tests(double[] all_vals) {
    if ((all_vals == null) || (all_vals.length <= 1))
      Dataset.perror("Feature.class :: <= 1 observed values for continuous feature");

    Vector v = new Vector();
    int i;

    minn = all_vals[0];
    maxx = all_vals[all_vals.length - 1];

    for (i = 0; i < all_vals.length - 1; i++) {
      if (all_vals[i] == all_vals[i + 1])
        Dataset.perror("Feature.class :: repeated feature values");
      v.addElement(new Double((all_vals[i] + all_vals[i + 1]) / 2.0));
    }
    tests = v;
  }

  public String tests() {
    int i, j;
    String v = "{";
    Vector dv;
    for (i = 0; i < tests.size(); i++) {
      if (Feature.IS_CONTINUOUS(type)) v += DF6.format(((Double) tests.elementAt(i)).doubleValue());
      else if (Feature.HAS_MODALITIES(type)) {
        dv = ((Vector) tests.elementAt(i));
        for (j = 0; j < dv.size(); j++) v += ((String) dv.elementAt(j));
      }
      if (i < tests.size() - 1) v += ", ";
    }
    v += "}";
    return v;
  }

  public static boolean IS_CONTINUOUS(String t) {
    return (t.equals(Feature.CONTINUOUS));
  }

  public static boolean IS_CLASS(String t) {
    return (t.equals(Feature.CLASS));
  }

  static int INDEX(String t) {
    int i = 0;
    do {
      if (t.equals(TYPE[i])) return i;
      i++;
    } while (i < TYPE.length);
    Dataset.perror("No type found for " + t);
    return -1;
  }

  static boolean HAS_MODALITIES(String t) {
    // synonym of is nominal

    return MODALITIES[Feature.INDEX(t)];
  }

  Feature(String n, String t, Vector m) {
    name = n;
    type = t;
    modalities = null;

    if (Feature.HAS_MODALITIES(t)) modalities = m;

    tests = Feature.TEST_LIST(this);
  }

  public boolean has_in_range(String s) {
    if (Feature.IS_CONTINUOUS(type))
      Dataset.perror(
          "Feature.class :: Continuous feature " + this + " queried for nominal value " + s);
    if (!Feature.HAS_MODALITIES(type))
      Dataset.perror("Feature.class :: feature type " + type + " unregistered ");

    int i;
    String ss;
    for (i = 0; i < modalities.size(); i++) {
      ss = (String) modalities.elementAt(i);
      if (ss.equals(s)) return true;
    }
    return false;
  }

  public String range() {
    String v = "";
    int i;
    if (Feature.HAS_MODALITIES(type)) {
      v += "{";
      for (i = 0; i < modalities.size(); i++) {
        v += "" + modalities.elementAt(i);
        if (i < modalities.size() - 1) v += ", ";
      }
      v += "}";
    } else if (Feature.IS_CONTINUOUS(type)) {
      v += "[" + minn + ", " + maxx + "]";
    }
    return v;
  }

  public boolean example_goes_left(Example e, int index_feature_in_e, int index_test) {
    // continuous values : <= is left, > is right
    // nominal values : in the set is left, otherwise is right

    double cv, tv;
    String nv;
    Vector ssv;
    int i;

    if (Feature.IS_CONTINUOUS(type)) {
      if (e.typed_features.elementAt(index_feature_in_e).getClass().getName().equals("String"))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a Double");

      cv = ((Double) e.typed_features.elementAt(index_feature_in_e)).doubleValue();
      tv = ((Double) tests.elementAt(index_test)).doubleValue();
      if (cv <= tv) return true;
      return false;
    } else if (Feature.HAS_MODALITIES(type)) {
      if (e.typed_features.elementAt(index_feature_in_e).getClass().getName().equals("Double"))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a String");

      nv = ((String) e.typed_features.elementAt(index_feature_in_e));
      ssv = (Vector) tests.elementAt(index_test);
      for (i = 0; i < ssv.size(); i++) {
        if (nv.equals((String) ssv.elementAt(i))) return true;
      }
      return false;
    } else Dataset.perror("Feature.class :: no type available for feature " + this);

    return false;
  }

  public String display_test(int index_test) {
    String v = name;
    int i;
    Vector ssv;

    if (Feature.IS_CONTINUOUS(type))
      v += " <= " + DF.format(((Double) tests.elementAt(index_test)).doubleValue());
    else if (Feature.HAS_MODALITIES(type)) {
      v += " in {";
      ssv = (Vector) tests.elementAt(index_test);
      for (i = 0; i < ssv.size(); i++) {
        v += (String) ssv.elementAt(i);
        if (i < ssv.size() - 1) v += ", ";
      }
      v += "}";
    } else Dataset.perror("Feature.class :: no type available for feature " + this);
    return v;
  }

  public String toString() {
    String v = "";
    int i;
    v += name + " -- " + type + " in " + range() + " -- tests : " + tests();

    return v;
  }
}
