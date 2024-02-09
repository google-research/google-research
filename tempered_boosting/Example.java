import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Example
 *****/

class Example implements Debuggable {
  int domain_id;

  double current_boosting_weight;
  // loss = @TemperedLoss => tempered weight
  // loss = @LogLoss => unnormalized boosting weight

  Vector typed_features; // features with type, excluding class
  double initial_class;

  double unnormalized_class;

  double normalized_class;
  double noisy_normalized_class;

  // class, only for training and in the context of LS model

  public void affiche() {
    System.out.println(this);
  }

  public String toString() {
    String v = "";
    int i;
    v += domain_id + " typed features : ";
    for (i = 0; i < typed_features.size(); i++) v += typed_features.elementAt(i) + " ";
    v +=
        " -> "
            + normalized_class
            + "("
            + noisy_normalized_class
            + "), w = "
            + current_boosting_weight;
    return v;
  }

  static Vector TO_TYPED_FEATURES(Vector ev, int index_class, Vector fv) {
    Vector vv = new Vector();
    int i;
    Feature f;

    for (i = 0; i < fv.size(); i++) {
      if (i != index_class) {
        f = (Feature) fv.elementAt(i);
        if (f.type.equals(Feature.CONTINUOUS))
          vv.addElement(new Double(Double.parseDouble((String) ev.elementAt(i))));
        else if (f.type.equals(Feature.NOMINAL))
          vv.addElement(new String((String) ev.elementAt(i)));
      }
    }

    return vv;
  }

  static double VAL_CLASS(Vector ev, int index_class) {
    return Double.parseDouble(((String) ev.elementAt(index_class)));
  }

  Example(int id, Vector v, int index_class, Vector fv) {
    domain_id = id;
    typed_features = Example.TO_TYPED_FEATURES(v, index_class, fv);
    initial_class = Example.VAL_CLASS(v, index_class);
    unnormalized_class = initial_class;

    current_boosting_weight = -1.0;
  }

  public int checkFeatures(Vector fv, int index_class) {
    // check that the example has features in the domain, otherwise errs

    int i, index = 0, vret = 0;
    Feature f;
    String fn;
    double fd;

    for (i = 0; i < fv.size(); i++) {
      if (i != index_class) {
        f = (Feature) fv.elementAt(i);
        if (f.type.equals(Feature.NOMINAL)) {
          fn = (String) typed_features.elementAt(index);
          if (!f.has_in_range(fn)) {
            Dataset.warning(
                "Example.class :: nominal attribute value "
                    + fn
                    + " not in range "
                    + f.range()
                    + " for feature "
                    + f.name);
            vret++;
          }
        }
        index++;
      }
    }
    return vret;
  }

  public void complete_normalized_class(
      double translate_v, double min_v, double max_v, double eta) {
    // TO Expand for more choices

    if (max_v != min_v) {
      if ((Dataset.DEFAULT_INDEX_FIT_CLASS != 0)
          && (Dataset.DEFAULT_INDEX_FIT_CLASS != 3)
          && (Dataset.DEFAULT_INDEX_FIT_CLASS != 4))
        Dataset.perror(
            "Example.class :: Choice "
                + Dataset.DEFAULT_INDEX_FIT_CLASS
                + " not implemented to fit the class");

      normalized_class =
          Dataset.TRANSLATE_SHRINK(
              unnormalized_class, translate_v, min_v, max_v, Feature.MAX_CLASS_MAGNITUDE);
      if (Dataset.DEFAULT_INDEX_FIT_CLASS == 4) normalized_class = Math.signum(normalized_class);
      unnormalized_class = 0.0;
    } else normalized_class = max_v;

    noisy_normalized_class = normalized_class;
    if (Utils.R.nextDouble() < eta) noisy_normalized_class = -noisy_normalized_class;
  }

  public boolean is_positive_noisy() {
    if (noisy_normalized_class == 0.0) Dataset.perror("Example.class :: normalized class is zero");
    if (noisy_normalized_class < 0.0) return false;
    return true;
  }
}
