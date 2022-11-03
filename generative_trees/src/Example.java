// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Example
 *****/

class Example implements Debuggable {
  int domain_id;

  double unnormalized_weight, sample_normalized_weight;

  Vector typed_features; // features with type, *including* class if any in the observed sample

  boolean is_real; // true = class +1 for detecting real vs fake

  Generator_Node generating_leaf;
  // if example generated, points to the (THEN) leaf that generated the example
  // make sure this stays a leaf ?

  Discriminator_Node training_leaf;
  // when example is in training, records the leaf it reaches in the current DT
  // null only for test examples

  double local_density;
  // for generated examples only, stores the local density at the generating leaf

  public static Vector TO_TYPED_FEATURES(Vector ev, Vector fv) {
    Vector vv = new Vector();
    int i;
    Feature f;

    for (i = 0; i < fv.size(); i++) {
      f = (Feature) fv.elementAt(i);
      if (Unknown_Feature_Value.IS_UNKNOWN(new String((String) ev.elementAt(i))))
        vv.addElement(new Unknown_Feature_Value());
      else if (f.type.equals(Feature.CONTINUOUS)) {

        if (Double.parseDouble((String) ev.elementAt(i)) == (double) Feature.FORBIDDEN_VALUE)
          Dataset.perror(
              "Example.class :: Forbidden value " + Feature.FORBIDDEN_VALUE + " found in example");

        vv.addElement(new Double(Double.parseDouble((String) ev.elementAt(i))));
      } else if (f.type.equals(Feature.INTEGER)) {

        if (Integer.parseInt((String) ev.elementAt(i)) == Feature.FORBIDDEN_VALUE)
          Dataset.perror(
              "Example.class :: Forbidden value " + Feature.FORBIDDEN_VALUE + " found in example");

        vv.addElement(new Integer(Integer.parseInt((String) ev.elementAt(i))));
      } else if (f.type.equals(Feature.NOMINAL))
        vv.addElement(new String((String) ev.elementAt(i)));
    }

    return vv;
  }

  public static Example copyOf(Example e) {
    // partial copy for test purpose / imputation essentially
    Example fc = new Example();
    int i;

    fc.domain_id = e.domain_id;
    fc.unnormalized_weight = e.unnormalized_weight;
    fc.sample_normalized_weight = e.sample_normalized_weight;
    fc.is_real = e.is_real;

    fc.typed_features = new Vector();
    for (i = 0; i < e.typed_features.size(); i++) {
      if (FEATURE_IS_UNKNOWN(e, i)) fc.typed_features.addElement(new Unknown_Feature_Value());
      else if (e.typed_features.elementAt(i).getClass().getSimpleName().equals("String"))
        fc.typed_features.addElement(new String((String) e.typed_features.elementAt(i)));
      else if (e.typed_features.elementAt(i).getClass().getSimpleName().equals("Double"))
        fc.typed_features.addElement(new Double((Double) e.typed_features.elementAt(i)));
      else if (e.typed_features.elementAt(i).getClass().getSimpleName().equals("Integer"))
        fc.typed_features.addElement(new Integer((Integer) e.typed_features.elementAt(i)));
    }

    // shallow there
    fc.generating_leaf = e.generating_leaf;
    fc.training_leaf = e.training_leaf;

    return fc;
  }

  public static boolean FEATURE_IS_UNKNOWN(Example ee, int i) {
    if ((ee.typed_features == null) || (ee.typed_features.elementAt(i) == null))
      Dataset.perror("Example.class :: no features");

    if ((ee.typed_features.elementAt(i).getClass().getSimpleName().equals("String"))
        || (ee.typed_features.elementAt(i).getClass().getSimpleName().equals("Double"))
        || (ee.typed_features.elementAt(i).getClass().getSimpleName().equals("Integer")))
      return false;

    if (!ee.typed_features.elementAt(i).getClass().getSimpleName().equals("Unknown_Feature_Value"))
      Dataset.perror(
          "Example.class :: unknown feature class "
              + ee.typed_features.elementAt(i).getClass().getSimpleName());

    return true;
  }

  Example() {
    domain_id = -1;
    typed_features = null;
    is_real = false;
    generating_leaf = null;
    training_leaf = null;

    unnormalized_weight = 0.5;
    sample_normalized_weight = -1.0;

    local_density = -1.0;
  }

  Example(int id, Vector v, Vector fv, boolean is_real) {
    domain_id = id;
    typed_features = Example.TO_TYPED_FEATURES(v, fv);

    this.is_real = is_real;
    generating_leaf = null;
    training_leaf = null;

    unnormalized_weight = 0.5;
    sample_normalized_weight = -1.0;
    local_density = -1.0;
  }

  Example(int id, Vector v, Vector fv, boolean is_real, Generator_Node gn) {
    this(id, v, fv, is_real);

    if ((!is_real) && (gn == null))
      Dataset.perror("Example.class :: generating a fake example with null generating leaf");

    if ((is_real) && (gn != null))
      Dataset.perror("Example.class :: generating a real example with NON-null generating leaf");

    generating_leaf = gn;
  }

  public boolean contains_unknown_values() {
    int i;
    for (i = 0; i < typed_features.size(); i++)
      if (Unknown_Feature_Value.IS_UNKNOWN(typed_features.elementAt(i))) return true;
    return false;
  }

  public String toString() {
    String v = "";
    int i;
    v += "#" + domain_id + ": ";
    for (i = 0; i < typed_features.size(); i++)
      if (Example.FEATURE_IS_UNKNOWN(this, i)) v += "[?] ";
      else v += typed_features.elementAt(i) + " ";

    if (!is_real) v += " ( <= #" + generating_leaf.name + " )";

    v += "\n";
    return v;
  }

  public String toStringSave() {
    String v = "";
    int i;
    for (i = 0; i < typed_features.size(); i++) {
      if (Example.FEATURE_IS_UNKNOWN(this, i)) v += Unknown_Feature_Value.S_UNKNOWN;
      else v += typed_features.elementAt(i);
      if (i < typed_features.size() - 1)
        v += Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX];
    }
    return v;
  }

  public String toStringSaveDensity(int x, int y) {
    String v = "";

    if (Example.FEATURE_IS_UNKNOWN(this, x)) v += Unknown_Feature_Value.S_UNKNOWN;
    else v += typed_features.elementAt(x);
    v += Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX];
    if (Example.FEATURE_IS_UNKNOWN(this, y)) v += Unknown_Feature_Value.S_UNKNOWN;
    else v += typed_features.elementAt(y);
    v += Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX];
    v += local_density;
    return v;
  }

  public void update_training_leaf(Discriminator_Node training_leaf) {
    if (is_real) this.training_leaf = training_leaf;
    else
      Dataset.perror(
          "Example.class :: updating training leaf not authorized for generated examples");
  }

  public void compute_sample_normalized_weight(double total) {
    if (total <= 0.0) Dataset.perror("Example.class :: total weight = " + total);
    sample_normalized_weight = unnormalized_weight / total;
  }

  public int checkAndCompleteFeatures(Vector fv) {
    // check that the example has features in the domain of each feature, otherwise errs

    int i, vret = 0;
    Feature f;
    String fn;
    double fd;
    int id;

    for (i = 0; i < fv.size(); i++) {
      f = (Feature) fv.elementAt(i);
      if (!typed_features.elementAt(i).getClass().getSimpleName().equals("Unknown_Feature_Value")) {
        if (f.type.equals(Feature.CONTINUOUS)) {
          fd = ((Double) typed_features.elementAt(i)).doubleValue();
          if (!f.has_in_range(fd)) {
            Dataset.warning(
                "Example.class :: continuous attribute value "
                    + fd
                    + " not in range "
                    + f.range(false)
                    + " for feature "
                    + f.name);
            vret++;
          }

          if ((f.dmin_from_data == f.dmax_from_data)
              && (f.dmin_from_data == (double) Feature.FORBIDDEN_VALUE)) {
            f.dmin_from_data = f.dmax_from_data = fd;
          } else if (f.dmin_from_data > fd) f.dmin_from_data = fd;
          else if (f.dmax_from_data < fd) f.dmax_from_data = fd;

        } else if (f.type.equals(Feature.INTEGER)) {
          id = ((Integer) typed_features.elementAt(i)).intValue();
          if (!f.has_in_range(id)) {
            Dataset.warning(
                "Example.class :: integer attribute value "
                    + id
                    + " not in range "
                    + f.range(false)
                    + " for feature "
                    + f.name);
            vret++;
          }

          if ((f.imin_from_data == f.imax_from_data)
              && (f.imin_from_data == Feature.FORBIDDEN_VALUE)) {
            f.imin_from_data = f.imax_from_data = id;
          } else if (f.imin_from_data > id) f.imin_from_data = id;
          else if (f.imax_from_data < id) f.imax_from_data = id;

        } else if (f.type.equals(Feature.NOMINAL)) {
          fn = (String) typed_features.elementAt(i);
          if (!f.has_in_range(fn)) {
            Dataset.warning(
                "Example.class :: nominal attribute value "
                    + fn
                    + " not in range "
                    + f.range(false)
                    + " for feature "
                    + f.name);
            vret++;
          }
        }
      }
    }
    return vret;
  }

  public boolean is_positive() {
    return is_real;
  }
}
