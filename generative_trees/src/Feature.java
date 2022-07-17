// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Unknown_Feature_Value: just to force errors if features not dealt properly during program
 *****/

class Unknown_Feature_Value {
  public static String S_UNKNOWN = "-1";

  public static boolean IS_UNKNOWN(double d) {
    String s = "" + d;
    return s.equals(S_UNKNOWN);
  }

  public static boolean IS_UNKNOWN(int i) {
    String s = "" + i;
    return s.equals(S_UNKNOWN);
  }

  public static boolean IS_UNKNOWN(String s) {
    return s.equals(S_UNKNOWN);
  }

  public static boolean IS_UNKNOWN(Object o) {
    if (o.getClass().getSimpleName().equals("Unknown_Feature_Value")) return true;
    return false;
  }

  public String toString() {
    return "Unknown Feature Value";
  }
}

/**************************************************************************************************************************************
 * Class Histogram: stores histograms for features
 *****/

class Histogram implements Debuggable {
  static int NUMBER_CONTINUOUS_FEATURE_BINS = 19;
  // make this an option -- choose a number that has little chance to produce observed values at the
  // boundary (e.g. large prime)

  static int MAX_NUMBER_INTEGER_FEATURE_BINS = 19;
  // make this an option
  // IF f.imax - f.imin + 1 > NUMBER_INTEGER_FEATURE_BINS, bins the histogram with bins of same size
  // up to 1

  String type, name;

  int feature_index;
  // reference index as in the dataset.features

  Vector<Feature> histogram_features;
  // feature vector for the histogram

  // IF NOMINAL: each feature has ONE modality for the bin
  // IF INTEGER: each feature has imin = imax = ONE modality
  // IF CONTINUOUS: each feature covers an interval [dmin, dmax] : NOTE closedness of intervals, not
  // an issue from measure standpoint

  Vector<Double> histogram_proportions;
  // in 1:1 wrt histogram_features

  Vector<Integer> histogram_counts;
  // in 1:1 wrt histogram_features

  Histogram() {
    type = name = "";
    feature_index = -1;
    histogram_features = null;
    histogram_proportions = null;
    histogram_counts = null;
  }

  Histogram(int find, Feature f) {
    // creates an empty histogram: initialise the histogram_proportions to 0.0

    feature_index = find;
    int i, index, ibinsize, ibinmin, ibinmax;
    double delta, be, en, local_eps;
    Vector<String> dumm;
    int[] binsize;
    histogram_features = new Vector<>();
    histogram_proportions = new Vector<>();
    histogram_counts = new Vector<>();

    type = f.type;
    name = f.name;

    if (Feature.IS_NOMINAL(f.type)) {
      if ((f.modalities == null) || (f.modalities.size() == 0))
        Dataset.perror("Histogram.class :: cannot create histogram for feature " + f);
      for (i = 0; i < f.modalities.size(); i++) {
        dumm = new Vector<>();
        dumm.addElement(f.modalities.elementAt(i));
        histogram_features.addElement(
            new Feature(f.name + "_" + i, f.type, dumm, f.dmin, f.dmax, false));
      }
    } else if (Feature.IS_CONTINUOUS(f.type)) {
      if (f.dmax == f.dmin)
        Dataset.perror("Histogram.class :: cannot create histogram for feature " + f);

      delta = (f.dmax - f.dmin) / (double) NUMBER_CONTINUOUS_FEATURE_BINS;

      be = f.dmin;

      for (i = 0; i < NUMBER_CONTINUOUS_FEATURE_BINS; i++) {
        en = be + delta;
        if (i == NUMBER_CONTINUOUS_FEATURE_BINS - 1) en = f.dmax;

        histogram_features.addElement(
            new Feature(f.name + "_" + i, f.type, f.modalities, be, en, false));
        be = en;
      }
    } else if (Feature.IS_INTEGER(f.type)) {
      if (f.imax - f.imin + 1 > MAX_NUMBER_INTEGER_FEATURE_BINS)
        ibinsize = MAX_NUMBER_INTEGER_FEATURE_BINS;
      else ibinsize = f.imax - f.imin + 1;

      binsize = new int[ibinsize];

      index = 0;
      for (i = 0; i < f.imax - f.imin + 1; i++) {
        binsize[index]++;
        index += 1;
        if (index > ibinsize - 1) index = 0;
      }

      ibinmin = f.imin;
      for (i = 0; i < ibinsize; i++) {
        ibinmax = ibinmin + binsize[i] - 1;
        histogram_features.addElement(
            new Feature(
                f.name + "_" + i, f.type, f.modalities, (double) ibinmin, (double) ibinmax, false));
        ibinmin = ibinmax + 1;
      }
    }

    for (i = 0; i < histogram_features.size(); i++) {
      histogram_proportions.addElement(new Double(0.0));
      histogram_counts.addElement(new Integer(0));
    }
  }

  public static Histogram copyOf(Histogram href) {
    Histogram ret = new Histogram();

    ret.feature_index = href.feature_index;
    ret.type = href.type;
    ret.name = href.name;

    ret.histogram_features = new Vector<>();
    ret.histogram_proportions = new Vector<>();
    ret.histogram_counts = new Vector<>();

    int i;

    for (i = 0; i < href.histogram_features.size(); i++) {
      ret.histogram_features.addElement(
          Feature.copyOf(href.histogram_features.elementAt(i), false));
      ret.histogram_proportions.addElement(
          new Double(href.histogram_proportions.elementAt(i).doubleValue()));
      ret.histogram_counts.addElement(new Integer(href.histogram_counts.elementAt(i).intValue()));
    }
    return ret;
  }

  public int index_bin(Example e) {
    // returns the index of the bin the example fills in (DOES NOT USE UNKNOWN FEATURES IF ANY)

    int i;
    int index = -1; // -1 = EXAMPLES HAS UNKNOWN FEATURE VALUE
    int number_satisfied = 0;
    if (!Example.FEATURE_IS_UNKNOWN(e, feature_index)) { // written for checks, can be sped up
      for (i = 0; i < histogram_proportions.size(); i++) {
        if (Feature.EXAMPLE_MATCHES_FEATURE(
            e,
            histogram_features.elementAt(i),
            feature_index,
            histogram_features.elementAt(i).type)) {
          if ((index == -1) || (Algorithm.R.nextDouble() < 0.5)) {
            index = i;
            number_satisfied++;
          }
        }
      }
      if (index == -1) {
        System.out.println(e + " :: ");
        for (i = 0; i < histogram_proportions.size(); i++)
          System.out.println(histogram_features.elementAt(i));

        Dataset.perror("Histogram.class :: example " + e + " has no bin in histogram ");
      }
    }
    return index;
  }

  public double percentage_intersection(int bin_index, Feature f) {
    if (!name.equals(f.name))
      Dataset.perror(
          "Feature.class :: name mismatch to compute probability_vector ("
              + name
              + " != "
              + f.name
              + ")");

    if (!type.equals(f.type))
      Dataset.perror(
          "Feature.class :: type mismatch to compute probability_vector ("
              + type
              + " != "
              + f.type
              + ")");

    double p = -1.0, i, r, l;

    if (Feature.IS_NOMINAL(f.type)) {
      if (f.modalities.contains(
          (String) histogram_features.elementAt(bin_index).modalities.elementAt(0)))
        p = 1.0 / ((double) f.modalities.size());
      else p = 0.0;
    } else if (Feature.IS_INTEGER(f.type)) {
      int tmin, tmax, fmin, fmax;
      tmin = histogram_features.elementAt(bin_index).imin;
      tmax = histogram_features.elementAt(bin_index).imax;
      fmin = f.imin;
      fmax = f.imax;

      if ((tmax < fmin) || (fmax < tmin)) return 0.0;

      if (fmax > tmax) r = (double) tmax;
      else r = (double) fmax;

      if (fmin < tmin) l = (double) tmin;
      else l = (double) fmin;

      p = (r - l + 1.0) / ((double) fmax - fmin + 1);

    } else if (Feature.IS_CONTINUOUS(f.type)) {
      double tmin, tmax, fmin, fmax;
      tmin = histogram_features.elementAt(bin_index).dmin;
      tmax = histogram_features.elementAt(bin_index).dmax;
      fmin = f.dmin;
      fmax = f.dmax;

      if ((tmax <= fmin) || (fmax <= tmin)) return 0.0;

      if (fmax > tmax) r = tmax;
      else r = fmax;

      if (fmin < tmin) l = tmin;
      else l = fmin;

      p = (r - l) / (fmax - fmin);
    }
    return p;
  }

  public double[] normalized_domain_intersection_with(Feature f) {
    // returns a vector WHOSE composants sum to 1.0,
    // counting the per-bin % intersection with 'f.domain'

    int i;
    double[] ret = new double[histogram_features.size()];
    double tot = 0.0;
    for (i = 0; i < histogram_features.size(); i++) {
      ret[i] = percentage_intersection(i, f);
      tot += ret[i];
    }

    if ((tot != 1.0) && (!(Statistics.APPROXIMATELY_EQUAL(tot, 1.0, EPS2))))
      Dataset.perror(
          "Histogram.class :: domain intersection not unit but = "
              + tot
              + " for feature "
              + f
              + " vs histogram "
              + this.toStringSave());
    return ret;
  }

  public boolean fill_histogram(Vector<Example> v, boolean use_weights) {
    // fills histogram
    // returns true IFF found at least one example with non-UNKNOWN value for feature

    int i, index, count;
    double tot_weight = 0.0, eweight, oldw;
    Example e;
    boolean ok = false;
    for (i = 0; i < v.size(); i++) {
      e = v.elementAt(i);
      index = index_bin(e);
      if (index != -1) {
        if (use_weights) eweight = e.unnormalized_weight;
        else eweight = 1.0;

        oldw = histogram_proportions.elementAt(index).doubleValue();
        count = histogram_counts.elementAt(index).intValue();

        oldw += eweight;
        count++;

        histogram_proportions.setElementAt(new Double(oldw), index);
        histogram_counts.setElementAt(new Integer(count), index);

        tot_weight += eweight;
        ok = true;
      }
    }

    if (tot_weight == 0.0)
      Dataset.perror("Histogram.class :: no weight counted in filling histogram ");
    for (i = 0; i < histogram_proportions.size(); i++) {
      oldw = histogram_proportions.elementAt(i).doubleValue();
      histogram_proportions.setElementAt(new Double(oldw / tot_weight), i);
    }
    return ok;
  }

  public void checkNormalized() {
    // check the histogram is normalized
    double tot = 0.0;
    int i;
    for (i = 0; i < histogram_proportions.size(); i++)
      tot += histogram_proportions.elementAt(i).doubleValue();
    if (!Statistics.APPROXIMATELY_EQUAL(tot, 1.0, EPS))
      Dataset.perror("Histogram.class :: not normalized ");
  }

  public String bucket_labels() {
    String v = "";
    int i;

    if (Feature.IS_NOMINAL(type)) {
      for (i = 0; i < histogram_features.size(); i++) {
        v += (histogram_features.elementAt(i)).modalities.elementAt(0);
        v += "\t(std_dev)";
        if (i < histogram_features.size() - 1) v += "\t";
      }
    } else if (Feature.IS_CONTINUOUS(type)) {
      for (i = 0; i < histogram_features.size(); i++) {
        v +=
            "["
                + DF6.format((histogram_features.elementAt(i)).dmin)
                + ","
                + DF6.format((histogram_features.elementAt(i)).dmax)
                + "]";
        v += "\t(std_dev)";
        if (i < histogram_features.size() - 1) v += "\t";
      }
    } else if (Feature.IS_INTEGER(type)) {
      for (i = 0; i < histogram_features.size(); i++) {
        v += "{" + (histogram_features.elementAt(i)).imin;
        if ((histogram_features.elementAt(i)).imax == (histogram_features.elementAt(i)).imin + 1)
          v += ", " + (histogram_features.elementAt(i)).imax;
        else if ((histogram_features.elementAt(i)).imax
            > (histogram_features.elementAt(i)).imin + 1)
          v += ",... " + (histogram_features.elementAt(i)).imax;
        v += "}";
        v += "\t(std_dev)";
        if (i < histogram_features.size() - 1) v += "\t";
      }
    }
    return v;
  }

  public String bucket_proportions() {
    String v = "";
    int i;

    for (i = 0; i < histogram_proportions.size(); i++) {
      v += "" + DF6.format(histogram_proportions.elementAt(i).doubleValue());
      v += "\t0.0";
      if (i < histogram_proportions.size() - 1) v += "\t";
    }

    return v;
  }

  public String binsToString() {
    String v = "";
    int i;
    v += Dataset.KEY_COMMENT + name + " (" + type + ") ";
    if (Feature.IS_CONTINUOUS(type)) v += "[#bins:" + NUMBER_CONTINUOUS_FEATURE_BINS + "]";
    else if (Feature.IS_INTEGER(type)) v += "{max #bins:" + MAX_NUMBER_INTEGER_FEATURE_BINS + "}";
    v += ":\n";
    v += Dataset.KEY_COMMENT;
    v += bucket_labels();
    v += "\n";

    return v;
  }

  public String toStringSave() {
    String v = "";
    int i;
    checkNormalized();
    v += Dataset.KEY_COMMENT + name + " (" + type + ") ";
    if (Feature.IS_CONTINUOUS(type)) v += "[#bins:" + NUMBER_CONTINUOUS_FEATURE_BINS + "]";
    else if (Feature.IS_INTEGER(type)) v += "{max #bins:" + MAX_NUMBER_INTEGER_FEATURE_BINS + "}";
    v += ":\n";
    v += Dataset.KEY_COMMENT;
    v += bucket_proportions();
    v += "\n";
    v += Dataset.KEY_COMMENT;
    v += bucket_labels();
    v += "\n";

    return v;
  }
}

/**************************************************************************************************************************************
 * Class Feature
 *****/

class Feature implements Debuggable {

  // All purpose variables (Discriminator and Generator)
  public static String NOMINAL = "NOMINAL", CONTINUOUS = "CONTINUOUS", INTEGER = "INTEGER";

  public static String TYPE[] = {Feature.NOMINAL, Feature.CONTINUOUS, Feature.INTEGER};

  public static int TYPE_INDEX(String s) {
    int i = 0;
    while (i < TYPE.length) {
      if (TYPE[i].equals(s)) return i;
      i++;
    }
    return -1;
  }

  public static String DISPERSION_NAME[] = {"Entropy", "Variance", "Variance"};

  // Discriminator relevant variables
  public static int NUMBER_CONTINUOUS_TIES = 100;
  // splits the interval in this number of internal splits (results in N+1 subintervals) FOR
  // CONTINUOUS VARIABLES

  public static int FORBIDDEN_VALUE = -100000;
  // value to initialise doubles and int. Must not be in dataset

  public static boolean DISPLAY_TESTS = false;

  // All purpose variables
  String name;
  String type;

  // Feature specific domain stuff -- redesign w/ a specific class for Generics
  Vector<String> modalities; // applies only to Feature.NOMINAL features
  double dmin, dmax; // applies only to Feature.CONTINUOUS features
  int imin, imax; // applies only to Feature.INTEGER features

  double dispertion_statistic_value;
  // Entropy for nominal, variance for ordered

  boolean formatted_for_a_dt;

  double[] dsplits_from_training; // COMPUTED
  int dmin_index_in_dsplits_from_training; // index of smallest split >= dmin
  int dmax_index_in_dsplits_from_training; // index of largest split <= dmax

  int[] isplits_from_training;
  int imin_index_in_dsplits_from_training; // index of smallest split >= imin
  int imax_index_in_dsplits_from_training; // index of largest split <= imax

  double dmin_from_data, dmax_from_data; // applies only to Feature.CONTINUOUS features
  int imin_from_data, imax_from_data; // applies only to Feature.INTEGER features

  // Generator relevant variables
  boolean empty_domain;
  // for Generator: true iff the feature can be used for generation at a given node

  // Discriminator relevant variables
  Vector tests;
  // set of test values as returned by TEST_LIST

  public static String SAVE_FEATURE(Feature f) {
    String ret = "";
    ret += f.name + "\t" + f.type + "\t";
    int i;

    if (Feature.IS_NOMINAL(f.type)) {
      if (f.modalities == null) ret += "null";
      else if (f.modalities.size() == 0) ret += "{}";
      else
        for (i = 0; i < f.modalities.size(); i++) {
          ret += (String) f.modalities.elementAt(i);
          if (i < f.modalities.size() - 1) ret += "\t";
        }
    } else if (Feature.IS_CONTINUOUS(f.type)) {
      ret += f.dmin + "\t" + f.dmax;
      if (f.dmin_from_data != f.dmax_from_data)
        ret += "\t" + f.dmin_from_data + "\t" + f.dmax_from_data;
    } else if (Feature.IS_INTEGER(f.type)) {
      ret += f.imin + "\t" + f.imax;
      if (f.imin_from_data != f.imax_from_data)
        ret += "\t" + f.imin_from_data + "\t" + f.imax_from_data;
    }
    return ret;
  }

  public double length() {
    double l = -1.0;
    if (Feature.IS_CONTINUOUS(type)) l = dmax - dmin;
    else if (Feature.IS_INTEGER(type)) l = (double) imax - (double) imin + 1.0;
    else if (Feature.IS_NOMINAL(type)) l = (double) modalities.size();

    if (l < 0.0) Dataset.perror("Feature.class :: feature " + this + " has <0 length");

    return l;
  }

  public boolean equals(Object o) {
    // makes sure the tests are ALSO in the SAME ORDER in both Features

    int i, j;

    if (o == this) return true;
    if (!(o instanceof Feature)) return false;
    Feature f = (Feature) o;
    if (!((((Feature.IS_NOMINAL(f.type)) && (Feature.IS_NOMINAL(type)))
        || ((Feature.IS_INTEGER(f.type)) && (Feature.IS_INTEGER(type)))
        || ((Feature.IS_CONTINUOUS(f.type)) && (Feature.IS_CONTINUOUS(type)))))) return false;

    if ((f.dmin_from_data != dmin_from_data)
        || (f.dmax_from_data != dmax_from_data)
        || (f.dmin != dmin)
        || (f.dmax != dmax)
        || (f.imin_from_data != imin_from_data)
        || (f.imax_from_data != imax_from_data)
        || (f.imin != imin)
        || (f.imax != imax)) return false;

    if (Feature.IS_NOMINAL(f.type)) {
      Vector<String> vf_test, v_test;
      if (f.modalities.size() != modalities.size()) return false;
      for (i = 0; i < f.modalities.size(); i++)
        if (!((String) f.modalities.elementAt(i)).equals(modalities.elementAt(i))) return false;

      if (f.tests.size() != tests.size()) return false;
      for (i = 0; i < f.tests.size(); i++) {
        vf_test = (Vector<String>) f.tests.elementAt(i);
        v_test = (Vector<String>) tests.elementAt(i);
        if (vf_test.size() != v_test.size()) return false;
        for (j = 0; j < vf_test.size(); j++)
          if (!((String) vf_test.elementAt(j)).equals(v_test.elementAt(j))) return false;
      }
    } else if (Feature.IS_INTEGER(f.type)) {
      if (f.tests.size() != tests.size()) return false;
      for (i = 0; i < f.tests.size(); i++)
        if (((Integer) f.tests.elementAt(i)).intValue()
            != ((Integer) tests.elementAt(i)).intValue()) return false;
    } else if (Feature.IS_CONTINUOUS(f.type)) {
      if (f.tests.size() != tests.size()) return false;
      for (i = 0; i < f.tests.size(); i++)
        if (((Double) f.tests.elementAt(i)).doubleValue()
            != ((Double) tests.elementAt(i)).doubleValue()) return false;
    }

    return true;
  }

  public static Feature copyOf(Feature f, boolean compute_tests) {
    // compute_tests = save memory for nominal features if tests not necessary

    Vector<String> v = null;
    double miv = (double) Feature.FORBIDDEN_VALUE, mav = (double) Feature.FORBIDDEN_VALUE;

    if (Feature.IS_NOMINAL(f.type)) v = new Vector<String>(f.modalities);
    else if (Feature.IS_CONTINUOUS(f.type)) {
      miv = f.dmin;
      mav = f.dmax;
    } else if (Feature.IS_INTEGER(f.type)) {
      miv = f.imin;
      mav = f.imax;
    }

    Feature fn = new Feature(f.name, f.type, v, miv, mav, compute_tests);

    if (Feature.IS_CONTINUOUS(f.type)) {
      fn.dmin_from_data = f.dmin_from_data;
      fn.dmax_from_data = f.dmax_from_data;

      fn.imin = fn.imax = fn.imin_from_data = fn.imax_from_data = Feature.FORBIDDEN_VALUE;
    } else if (Feature.IS_INTEGER(f.type)) {
      fn.imin_from_data = f.imin_from_data;
      fn.imax_from_data = f.imax_from_data;

      fn.dmin = fn.dmax = fn.dmin_from_data = fn.dmax_from_data = Feature.FORBIDDEN_VALUE;
    }

    return fn;
  }

  public static Feature copyOf(
      Feature f,
      boolean compute_tests,
      boolean use_double_splits_from_training,
      Dataset ds,
      int f_index) {
    // make use this is not used for non CONTINUOUS VARIABLES or not for DT induction
    if (!use_double_splits_from_training) return copyOf(f, compute_tests);
    else {
      if (!Feature.IS_CONTINUOUS(f.type))
        Dataset.perror("Feature.class :: copyOf has to be sent for continuous variables");

      Feature fn = copyOf(f, compute_tests);
      // updates of tests related variables

      int i;
      if (f.tests == null) fn.tests = null;
      else {
        if (f.tests.size() == 0)
          Dataset.perror(
              "Feature.class :: zero test in a non-null test vector -- this is an error");
        fn.tests = new Vector<Double>();
        fn.dmin_index_in_dsplits_from_training = f.dmin_index_in_dsplits_from_training;
        fn.dmax_index_in_dsplits_from_training = f.dmax_index_in_dsplits_from_training;

        for (i = 0; i < f.tests.size(); i++)
          fn.tests.addElement(new Double(((Double) f.tests.elementAt(i)).doubleValue()));
      }
      return fn;
    }
  }

  public static boolean EXAMPLE_MATCHES_FEATURE(Example e, Feature f, int f_index, String f_type) {
    // checks whether e.typed_features.elementAt(f_index) is in domain of f
    if (e.typed_features
        .elementAt(f_index)
        .getClass()
        .getSimpleName()
        .equals("Unknown_Feature_Value")) return true;
    double ed;
    int ei;
    String es;

    if (f_type.equals(Feature.CONTINUOUS)) {
      if (!Feature.IS_CONTINUOUS(f.type))
        Dataset.perror("Feature.class :: feature type mismatch -- CONTINUOUS");

      ed = ((Double) e.typed_features.elementAt(f_index)).doubleValue();
      if ((ed >= f.dmin) && (ed <= f.dmax)) return true;
      else return false;
    } else if (f_type.equals(Feature.INTEGER)) {
      if (!Feature.IS_INTEGER(f.type))
        Dataset.perror("Feature.class :: feature type mismatch -- INTEGER");

      ei = ((Integer) e.typed_features.elementAt(f_index)).intValue();
      if ((ei >= f.imin) && (ei <= f.imax)) return true;
      else return false;
    } else if (f_type.equals(Feature.NOMINAL)) {
      if (!Feature.IS_NOMINAL(f.type))
        Dataset.perror("Feature.class :: feature type mismatch -- NOMINAL");

      es = (String) e.typed_features.elementAt(f_index);
      if (f.modalities.contains(es)) return true;
      else return false;
    } else Dataset.perror("Feature.class :: feature type unknown");

    return true;
  }

  public static double RELATIVE_PERCENTAGE_SUPPORT(Feature a, Feature b) {
    // returns size_domain(a \cap b) / size_domain(b)
    double num = 0.0, den = 0.0, dinf, dsup, rat = -1.0;
    int i;

    if (!a.type.equals(b.type)) Dataset.perror("Feature.class :: not the same class for features");

    if (Feature.IS_CONTINUOUS(a.type)) {
      if ((b.dmax == b.dmin) || (a.dmax <= b.dmin) || (a.dmin >= b.dmax)) return 0.0;

      if (a.dmin < b.dmin) dinf = b.dmin;
      else dinf = a.dmin;

      if (a.dmax > b.dmax) dsup = b.dmax;
      else dsup = a.dmax;

      num = dsup - dinf;
      den = b.dmax - b.dmin;
    } else if (Feature.IS_INTEGER(a.type)) {
      if ((a.imax < b.imin) || (a.imin > b.imax)) return 0.0;

      if (a.imin <= b.imin) dinf = (double) b.imin;
      else dinf = (double) a.imin;

      if (a.imax >= b.imax) dsup = (double) b.imax;
      else dsup = (double) a.imax;

      num = dsup - dinf + 1.0;
      den = ((double) (b.imax - b.imin)) + 1.0;
    } else if (Feature.IS_NOMINAL(a.type)) {
      if (b.modalities.size() == 0) return 0.0;
      else den = (double) b.modalities.size();

      num = 0.0;
      for (i = 0; i < a.modalities.size(); i++)
        if (b.modalities.contains((String) a.modalities.elementAt(i))) num += 1.0;
    } else Dataset.perror("Feature.class :: feature type unknown");

    rat = num / den;
    if ((rat < 0.0) || (rat > 1.0))
      Dataset.perror("Feature.class :: ratio " + rat + " not a probability");
    return rat;
  }

  public static double RELATIVE_PERCENTAGE_SUPPORT(
      Discriminator_Node dn_leaf, Generator_Tree gt, Generator_Node gn_leaf) {
    // Pr[dn_leaf | ~gn_leaf]
    // returns the relative percentage (wrt gn_leaf) of the support intersection with dn_leaf
    // do not forget to *= gn_leaf.local_p = Pr[~gn_leaf] for n_lambda

    if (!dn_leaf.is_leaf) {
      if ((!dn_leaf.left_child.is_pure()) && (!dn_leaf.right_child.is_pure()))
        Dataset.perror(
            "Feature.class :: Discriminator_Node #"
                + dn_leaf.name
                + " not a discriminator leaf BUT both children nodes not pure");
    }
    if (!gn_leaf.is_leaf) Dataset.perror("Feature.class :: not a generator leaf");
    Vector<Feature> vf_dn = dn_leaf.support_at_classification_node;
    Vector<Feature> vf_gn = Generator_Node.ALL_FEATURES_DOMAINS_AT_SAMPLING_NODE(gt, gn_leaf);
    int i;
    double v = 1.0;

    if (vf_dn.size() != vf_gn.size())
      Dataset.perror("Feature.class :: Feature vectors not of the same size");

    double[] all_rat = new double[vf_gn.size()];
    for (i = 0; i < vf_gn.size(); i++) {
      all_rat[i] = Feature.RELATIVE_PERCENTAGE_SUPPORT(vf_dn.elementAt(i), vf_gn.elementAt(i));
      v *= all_rat[i];
    }

    return v;
  }

  public static double RELATIVE_PERCENTAGE_SUPPORT(
      Discriminator_Node dn_leaf,
      Generator_Tree gt,
      Generator_Node gn_leaf,
      Feature sub_feature,
      int sub_feature_index) {
    // Pr[dn_leaf | ~gn_leaf \wedge X_[l|r]]
    // returns the relative percentage (wrt gn_leaf) of the support intersection with dn_leaf
    // do not forget to *= gn_leaf.local_p for n_lambda

    if (!dn_leaf.is_leaf) {
      if ((!dn_leaf.left_child.is_pure()) && (!dn_leaf.right_child.is_pure()))
        Dataset.perror(
            "Feature.class :: Discriminator_Node #"
                + dn_leaf.name
                + " not a discriminator leaf BUT both children nodes not pure");
    }
    if (!gn_leaf.is_leaf) Dataset.perror("Feature.class :: not a generator leaf");
    Vector<Feature> vf_dn = dn_leaf.support_at_classification_node;
    Vector<Feature> vf_gn = Generator_Node.ALL_FEATURES_DOMAINS_AT_SAMPLING_NODE(gt, gn_leaf);
    if (!IS_SUBFEATURE(sub_feature, vf_gn.elementAt(sub_feature_index)))
      Dataset.perror(
          "Feature.class :: "
              + sub_feature
              + " not a subfeature of "
              + vf_gn.elementAt(sub_feature_index));

    vf_gn.setElementAt(sub_feature, sub_feature_index);

    int i;
    double v = 1.0;

    if (vf_dn.size() != vf_gn.size())
      Dataset.perror("Feature.class :: Feature vectors not of the same size");

    double[] all_rat = new double[vf_gn.size()];
    for (i = 0; i < vf_gn.size(); i++) {
      all_rat[i] = Feature.RELATIVE_PERCENTAGE_SUPPORT(vf_dn.elementAt(i), vf_gn.elementAt(i));
      v *= all_rat[i];
    }

    return v;
  }

  public static boolean HAS_SINGLETON_DOMAIN(Feature f_node) {
    // such features MUST NOT be split (important when unknown feature values)
    if ((Feature.IS_INTEGER(f_node.type)) && (f_node.imin == f_node.imax)) return true;
    if (Feature.IS_NOMINAL(f_node.type)) {
      if (f_node.modalities == null) Dataset.perror("Feature.class :: untestable feature");
      if (f_node.modalities.size() == 0)
        Dataset.perror("Feature.class :: feature with empty domain");
      if (f_node.modalities.size() == 1) return true;
    }
    return false;
  }

  public static boolean SPLIT_AUTHORIZED(Feature f_split) {
    if (!Discriminator_Tree.USE_OBSERVED_FEATURE_VALUES_FOR_SPLITS) return true;

    if (!Feature.IS_CONTINUOUS(f_split.type)) return true;

    if ((f_split.dmin_index_in_dsplits_from_training <= f_split.dmax_index_in_dsplits_from_training)
        && (f_split.dmin_index_in_dsplits_from_training != -1)
        && (f_split.dmax_index_in_dsplits_from_training != -1)) return true;

    return false;
  }

  public static Vector<Feature>[] SPLIT_SUPPORT_DT_INDUCTION(
      Discriminator_Node dn, int feature_index, int test_index_in_feature_index, Dataset ds) {
    // returns two vectors, each being the full support of dn after its split on test_index w/
    // feature_index
    // WARNING: test_index_reference gives the test IN THE ROOT'S FEATURE'S SET ("reference", given
    // in args)
    // 0 = left, 1 = right;

    Vector<Feature>[] split = new Vector[2];
    split[0] = new Vector<Feature>();
    split[1] = new Vector<Feature>();
    Feature[] split_feature;
    int i, excluded_index = test_index_in_feature_index;
    for (i = 0; i < dn.support_at_classification_node.size(); i++) {
      if (i == feature_index) {
        split_feature =
            SPLIT_FEATURE(
                dn.support_at_classification_node.elementAt(i), test_index_in_feature_index, true);

        if ((Feature.IS_CONTINUOUS(dn.support_at_classification_node.elementAt(i).type))
            && (Discriminator_Tree.USE_OBSERVED_FEATURE_VALUES_FOR_SPLITS)) {
          if ((dn.support_at_classification_node.elementAt(i).dmin_index_in_dsplits_from_training
                  == -1)
              || (dn.support_at_classification_node.elementAt(i).dmax_index_in_dsplits_from_training
                  == -1)) Dataset.perror("Feature.class :: feature should not be splittable");

          if (test_index_in_feature_index > 0) {
            split_feature[0].dmin_index_in_dsplits_from_training =
                dn.support_at_classification_node.elementAt(i).dmin_index_in_dsplits_from_training;
            split_feature[0].dmax_index_in_dsplits_from_training =
                dn.support_at_classification_node.elementAt(i).dmin_index_in_dsplits_from_training
                    + test_index_in_feature_index
                    - 1;

            split_feature[0].try_format_tests(ds, i, false);
          } else {
            split_feature[0].dmin_index_in_dsplits_from_training =
                split_feature[0].dmax_index_in_dsplits_from_training = -1;
            split_feature[0].tests = null;
          }

          if (test_index_in_feature_index
              < dn.support_at_classification_node.elementAt(i).dmax_index_in_dsplits_from_training
                  - dn.support_at_classification_node.elementAt(i)
                      .dmin_index_in_dsplits_from_training) {
            split_feature[1].dmin_index_in_dsplits_from_training =
                dn.support_at_classification_node.elementAt(i).dmin_index_in_dsplits_from_training
                    + test_index_in_feature_index
                    + 1;
            split_feature[1].dmax_index_in_dsplits_from_training =
                dn.support_at_classification_node.elementAt(i).dmax_index_in_dsplits_from_training;

            split_feature[1].try_format_tests(ds, i, false);
          } else {
            split_feature[1].dmin_index_in_dsplits_from_training =
                split_feature[1].dmin_index_in_dsplits_from_training = -1;
            split_feature[1].tests = null;
          }

          CHECK_TESTS_DISJOINT_UNION(
              dn.support_at_classification_node.elementAt(i),
              split_feature[0],
              split_feature[1],
              excluded_index);
        }

        split[0].addElement((Feature) split_feature[0]);
        split[1].addElement((Feature) split_feature[1]);
      } else if (Feature.IS_CONTINUOUS(dn.support_at_classification_node.elementAt(i).type)) {
        split[0].addElement(
            Feature.copyOf(
                (Feature) dn.support_at_classification_node.elementAt(i),
                true,
                Discriminator_Tree.USE_OBSERVED_FEATURE_VALUES_FOR_SPLITS,
                ds,
                i));
        split[1].addElement(
            Feature.copyOf(
                (Feature) dn.support_at_classification_node.elementAt(i),
                true,
                Discriminator_Tree.USE_OBSERVED_FEATURE_VALUES_FOR_SPLITS,
                ds,
                i));
      } else {
        split[0].addElement(
            Feature.copyOf((Feature) dn.support_at_classification_node.elementAt(i), true));
        split[1].addElement(
            Feature.copyOf((Feature) dn.support_at_classification_node.elementAt(i), true));
      }
    }

    return split;
  }

  public static void TEST_UNION(Feature f_parent, Feature f_left, Feature f_right) {
    // controls the union of the children features is the parent
    // ensures the children have non-empty domain
    int i;

    if ((!f_parent.type.equals(f_left.type)) || (!f_parent.type.equals(f_right.type)))
      Dataset.perror(
          "Feature.class :: parent feature of type "
              + f_parent.type
              + " but children: "
              + f_left.type
              + ", "
              + f_right.type);

    if (Feature.IS_CONTINUOUS(f_parent.type)) {
      if ((f_left.dmin != f_parent.dmin) || (f_right.dmax != f_parent.dmax))
        Dataset.perror("Feature.class :: double domain does not cover parent's range");
      if (f_left.dmax != f_right.dmin)
        Dataset.perror("Feature.class :: double domain union mismatch");
    } else if (Feature.IS_INTEGER(f_parent.type)) {
      if ((f_left.imin != f_parent.imin) || (f_right.imax != f_parent.imax))
        Dataset.perror("Feature.class :: integer domain does not cover parent's range");
      if (f_left.imax + 1 != f_right.imin)
        Dataset.perror(
            "Feature.class :: integer domain union mismatch : f_left.imax = "
                + f_left.imax
                + ", f_right.imin = "
                + f_right.imin);
    } else if (Feature.IS_NOMINAL(f_parent.type)) {
      if ((f_left.modalities == null) || (f_right.modalities == null))
        Dataset.perror("Feature.class :: nominal domain has null domain in a child");
      if ((f_left.modalities.size() == 0) || (f_right.modalities.size() == 0))
        Dataset.perror("Feature.class :: nominal domain has empty domain in a child");
      if (f_parent.modalities == null)
        Dataset.perror("Feature.class :: nominal domain has null domain in parent");
      if (f_parent.modalities.size() == 0)
        Dataset.perror("Feature.class :: nominal domain has empty domain in parent");
      for (i = 0; i < f_left.modalities.size(); i++)
        if (!f_parent.modalities.contains((String) f_left.modalities.elementAt(i)))
          Dataset.perror(
              "Feature.class :: parent's nominal domain does not contain left child modality "
                  + ((String) f_left.modalities.elementAt(i)));
      for (i = 0; i < f_right.modalities.size(); i++)
        if (!f_parent.modalities.contains((String) f_right.modalities.elementAt(i)))
          Dataset.perror(
              "Feature.class :: parent's nominal domain does not contain right child modality "
                  + ((String) f_right.modalities.elementAt(i)));
      if (f_left.modalities.size() + f_right.modalities.size() != f_parent.modalities.size())
        Dataset.perror(
            "Feature.class :: parent's nominal domain contains modalities not in children");
    }
  }

  public static Feature[] SPLIT_FEATURE(Feature f, int test_index, boolean compute_tests) {
    // returns TWO features by applying f.tests.elementAt(test_index) to the domain of f
    // base_name used to name the features
    Feature[] ft = new Feature[2];
    Feature left = null, right = null;
    Vector<String> vright;
    int i, bound;

    if (Feature.IS_CONTINUOUS(f.type)) {
      left =
          new Feature(
              f.name,
              f.type,
              f.modalities,
              f.dmin,
              ((Double) f.tests.elementAt(test_index)).doubleValue(),
              compute_tests);
      right =
          new Feature(
              f.name,
              f.type,
              f.modalities,
              ((Double) f.tests.elementAt(test_index)).doubleValue(),
              f.dmax,
              compute_tests);
    } else if (Feature.IS_INTEGER(f.type)) {
      if (test_index < f.tests.size() - 1)
        bound =
            ((Integer) f.tests.elementAt(test_index + 1))
                .intValue(); // takes the next value in the test list
      else bound = f.imax; // just the max available

      left =
          new Feature(
              f.name,
              f.type,
              f.modalities,
              f.imin,
              ((Integer) f.tests.elementAt(test_index)).intValue(),
              compute_tests);
      right = new Feature(f.name, f.type, f.modalities, bound, f.imax, true);
    } else if (Feature.IS_NOMINAL(f.type)) {
      if (f.tests == null)
        Dataset.perror(
            "Feature.class :: no test for nominal feature " + f + " at index " + test_index);

      left =
          new Feature(
              f.name,
              f.type,
              (Vector<String>) f.tests.elementAt(test_index),
              f.dmin,
              f.dmax,
              compute_tests);
      vright = new Vector<String>();
      for (i = 0; i < f.modalities.size(); i++) {
        if (!((Vector) f.tests.elementAt(test_index)).contains((String) f.modalities.elementAt(i)))
          vright.addElement(new String((String) f.modalities.elementAt(i)));
      }
      if (vright.size() == 0)
        Dataset.perror("Feature.class :: no modality to add to the right split");
      right = new Feature(f.name, f.type, vright, f.dmin, f.dmax, compute_tests);
    }

    ft[0] = left;
    ft[1] = right;

    return ft;
  }

  public static boolean IS_SUBFEATURE(Feature a, Feature b) {
    // checks if domain(a) \subseteq domain(b)
    return IS_SUBFEATURE(a, -1, b, -1);
  }

  public static boolean IS_SUBFEATURE(Feature a, int index_a, Feature b, int index_b) {
    // checks if domain(a) \subseteq domain(b) AND returns an error if index_a != index_b (in
    // myDomain.myDS.features)
    // also checks inconsistencies: one of a or b must be a subfeature of the other AND the feature
    // type values must have been computed

    boolean anotinb, bnotina;
    int i, ia, ib;

    if (index_a != index_b)
      Dataset.perror("Feature.class :: not the same feature (" + index_a + " != " + index_b + ")");
    if (!a.type.equals(b.type))
      Dataset.perror(
          "Feature.class :: not the same type of feature (" + a.type + " != " + b.type + ")");

    if (IS_CONTINUOUS(a.type)) {
      if (a.dmin >= b.dmin) {
        if (a.dmax <= b.dmax) return true;
        else
          Dataset.perror(
              "Feature.class :: inconsistency for subfeature check for : ("
                  + a.dmin
                  + ", "
                  + a.dmax
                  + ") subseteq ("
                  + b.dmin
                  + ", "
                  + b.dmax
                  + ") ? ");
      } else if (a.dmax < b.dmax)
        Dataset.perror(
            "Feature.class :: inconsistency for subfeature check for : ("
                + a.dmin
                + ", "
                + a.dmax
                + ") subseteq ("
                + b.dmin
                + ", "
                + b.dmax
                + ") ? ");
    } else if (IS_INTEGER(a.type)) {
      if (a.imin >= b.imin) {
        if (a.imax <= b.imax) return true;
        else
          Dataset.perror(
              "Feature.class :: inconsistency for subfeature check for : ("
                  + a.imin
                  + ", "
                  + a.imax
                  + ") subseteq ("
                  + b.imin
                  + ", "
                  + b.imax
                  + ") ? ");
      } else if (a.imax < b.imax)
        Dataset.perror(
            "Feature.class :: inconsistency for subfeature check for : ("
                + a.imin
                + ", "
                + a.imax
                + ") subseteq ("
                + b.imin
                + ", "
                + b.imax
                + ") ? ");
    } else if (IS_NOMINAL(a.type)) {
      if (a.modalities == null) return true;
      else if (b.modalities != null) {
        anotinb = bnotina = false;
        ia = ib = -1;
        for (i = 0; i < a.modalities.size(); i++)
          if (!b.modalities.contains((String) a.modalities.elementAt(i))) {
            anotinb = true;
            ia = i;
          }
        for (i = 0; i < b.modalities.size(); i++)
          if (!a.modalities.contains((String) b.modalities.elementAt(i))) {
            bnotina = true;
            ib = i;
          }
        if ((anotinb) && (bnotina))
          Dataset.perror(
              "Feature.class :: inconsistency for subfeature check for : "
                  + ((String) a.modalities.elementAt(ia))
                  + " not in b and "
                  + ((String) b.modalities.elementAt(ib))
                  + " not in a ");
        else if (!anotinb) return true;
      }
    } else Dataset.perror("Feature.class :: no Feature type for " + a.type);

    return false;
  }

  public static Vector TEST_LIST(Feature f) {
    // if continuous, list of evenly spaced ties UNLESS
    // split_for_a_dt_and_use_observed_values_for_splits = true, in which case uses splits computed
    // from training sample
    // if nominal, list of partial non-empty subsets of the whole set
    // if integer, list of integers

    Vector v = new Vector();
    if (IS_CONTINUOUS(f.type)) {
      if (f.dmax - f.dmin <= 0.0) {
        v = null;
      } else {
        double vmin = f.dmin;
        double vmax = f.dmax;
        double delta = (vmax - vmin) / ((double) (NUMBER_CONTINUOUS_TIES + 1));
        double vcur = vmin + delta;
        int i;
        for (i = 0; i < NUMBER_CONTINUOUS_TIES; i++) {
          v.addElement(new Double(vcur));
          vcur += delta;
        }
      }
    } else if (IS_INTEGER(f.type)) {
      if (f.imax - f.imin <= 0) {
        v = null;
      } else {
        int vmin = f.imin;
        int nvals = f.imax - f.imin;
        int i;
        for (i = 0; i < nvals; i++) {
          v.addElement(new Integer(vmin + i));
        }
      }
    } else if (IS_NOMINAL(f.type)) {
      if (!Discriminator_Tree.RANDOMISE_SPLIT_FINDING_WHEN_TOO_MANY_SPLITS)
        v = Utils.ALL_NON_TRIVIAL_SUBSETS(f.modalities);
      else if (f.modalities.size() <= Discriminator_Tree.MAX_CARD_MODALITIES_BEFORE_RANDOMISATION)
        v = Utils.ALL_NON_TRIVIAL_SUBSETS(f.modalities);
      else
        v =
            Utils.ALL_NON_TRIVIAL_BOUNDED_SUBSETS(
                f.modalities, Discriminator_Tree.MAX_SIZE_FOR_RANDOMISATION);
      if (v.size() == 0) v = null;
    }

    return v;
  }

  public String tests(boolean show_all) {
    int max_display = 5;
    if (show_all) max_display = tests.size();

    int i, j;
    String v = "{";
    Vector dv;
    if (tests != null) {
      if (tests.size() == 0) Dataset.perror("Feature.class :: avoid empty but non null test sets");
      for (i = 0; i < tests.size(); i++) {
        if (i < max_display) {
          if (Feature.IS_CONTINUOUS(type))
            v += DF4.format(((Double) tests.elementAt(i)).doubleValue());
          else if (Feature.IS_INTEGER(type)) v += ((Integer) tests.elementAt(i)).intValue();
          else if (Feature.IS_NOMINAL(type)) {
            dv = ((Vector) tests.elementAt(i));
            for (j = 0; j < dv.size(); j++) v += ((String) dv.elementAt(j)) + " ";
          }
          if (i < tests.size() - 1) v += ", ";
        } else {
          v += "... ";
          break;
        }
      }
    }
    v += "}";
    return v;
  }

  public static boolean IS_CONTINUOUS(String t) {
    return (t.equals(Feature.CONTINUOUS));
  }

  public static boolean IS_INTEGER(String t) {
    // equiv. to Nominal Mono-valued, ordered

    return (t.equals(Feature.INTEGER));
  }

  static boolean IS_NOMINAL(String t) {
    // Nominal Mono-Valued, no order

    return (t.equals(Feature.NOMINAL));
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

  public static void CHECK_TESTS_DISJOINT_UNION(
      Feature parent, Feature f1, Feature f2, int excluded_index) {
    // various checks

    if ((!Discriminator_Tree.USE_OBSERVED_FEATURE_VALUES_FOR_SPLITS)
        || (!Feature.IS_CONTINUOUS(parent.type))
        || (!Feature.IS_CONTINUOUS(f1.type))
        || (!Feature.IS_CONTINUOUS(f2.type))) return;

    boolean[] found = new boolean[parent.tests.size()];
    boolean found_f = false, duplicate = false;
    int i, j, k;
    if (f1.tests != null)
      for (i = 0; i < f1.tests.size(); i++) {
        found_f = false;
        j = 0;
        do {
          if (((Double) f1.tests.elementAt(i)).doubleValue()
              == ((Double) parent.tests.elementAt(j)).doubleValue()) {
            found[j] = true;
            found_f = true;
          } else j++;
        } while ((!found_f) && (j < parent.tests.size()));
        if (!found_f) {
          System.out.print("\n Parent Tests : ");
          for (k = 0; k < parent.tests.size(); k++)
            System.out.print((Double) parent.tests.elementAt(k) + ", ");
          System.out.print("\n F1 Tests : ");
          for (k = 0; k < f1.tests.size(); k++)
            System.out.print((Double) f1.tests.elementAt(k) + ", ");

          Dataset.perror(
              " Feature.class :: test value "
                  + ((Double) f1.tests.elementAt(i)).doubleValue()
                  + " not found in parent");
        }
      }

    if (f2.tests != null)
      for (i = 0; i < f2.tests.size(); i++) {
        found_f = false;
        j = 0;
        do {
          if (((Double) f2.tests.elementAt(i)).doubleValue()
              == ((Double) parent.tests.elementAt(j)).doubleValue()) {
            if (found[j]) duplicate = true;
            else {
              found[j] = true;
              found_f = true;
            }
          } else j++;
        } while ((!found_f) && (!duplicate) && (j < parent.tests.size()));
        if (duplicate) {
          // something went wrong

          System.out.print("\n F1 Tests : ");
          for (k = 0; k < f1.tests.size(); k++)
            System.out.print((Double) f1.tests.elementAt(k) + ", ");
          System.out.print("\n F2 Tests : ");
          for (k = 0; k < f2.tests.size(); k++)
            System.out.print((Double) f2.tests.elementAt(k) + ", ");
          Dataset.perror(
              " Feature.class :: test value "
                  + ((Double) parent.tests.elementAt(j)).doubleValue()
                  + " duplicate in F1 and F2");
        }
        if (!found_f) {
          // something went wrong

          System.out.print("\n Excluded index : " + excluded_index);
          System.out.print("\n Parent Tests : ");
          for (k = 0; k < parent.tests.size(); k++)
            System.out.print((Double) parent.tests.elementAt(k) + ", ");
          if (f1.tests != null) {
            System.out.print("\n F1 Tests : ");
            for (k = 0; k < f1.tests.size(); k++)
              System.out.print((Double) f1.tests.elementAt(k) + ", ");
          }
          System.out.print("\n F2 Tests : ");
          for (k = 0; k < f2.tests.size(); k++)
            System.out.print((Double) f2.tests.elementAt(k) + ", ");

          Dataset.perror(
              " Feature.class :: test value "
                  + ((Double) f2.tests.elementAt(i)).doubleValue()
                  + " not found in parent");
        }
      }

    found_f = false;
    for (i = 0; i < found.length; i++) {
      if ((!found[i]) && (i != excluded_index)) {
        found_f = true;
        System.out.println((Double) parent.tests.elementAt(i) + " not found");
      }
    }
    if (found_f) System.exit(0);
  }

  public void init_splits(Dataset ds, int f_index, int cv_fold) {
    if (Feature.IS_NOMINAL(type))
      Dataset.perror("Feature.class :: bad call of init_splits for Nominal feature " + name);

    int i;
    Example e;
    if (Feature.IS_CONTINUOUS(type)) {
      Vector<Double> all_observed = new Vector<>();
      double[] all_observed_ordered;

      for (i = 0; i < ds.train_size(cv_fold, true); i++) {
        e = ds.train_example(cv_fold, i, true);
        if (!Example.FEATURE_IS_UNKNOWN(e, f_index)) {
          if (!e.typed_features.elementAt(f_index).getClass().getSimpleName().equals("Double"))
            Dataset.perror("Feature.class :: Example feature " + f_index + " not a double");
          all_observed.addElement(
              new Double(((Double) e.typed_features.elementAt(f_index)).doubleValue()));
        }
      }

      all_observed_ordered = new double[all_observed.size()];
      for (i = 0; i < all_observed.size(); i++)
        all_observed_ordered[i] = all_observed.elementAt(i).doubleValue();
      Arrays.sort(all_observed_ordered);

      all_observed = new Vector<>();
      for (i = 0; i < all_observed_ordered.length; i++)
        if ((i == 0) || (all_observed_ordered[i] != all_observed_ordered[i - 1]))
          all_observed.addElement(new Double(all_observed_ordered[i]));

      dsplits_from_training = new double[all_observed.size() - 1];
      for (i = 0; i < dsplits_from_training.length; i++)
        dsplits_from_training[i] =
            ((all_observed.elementAt(i).doubleValue())
                    + (all_observed.elementAt(i + 1).doubleValue()))
                / 2.0;
    } else if (Feature.IS_INTEGER(type)) {
      Vector<Integer> all_observed = new Vector<>();
      int[] all_observed_ordered;

      for (i = 0; i < ds.train_size(cv_fold, true); i++) {
        e = ds.train_example(cv_fold, i, true);
        if (!Example.FEATURE_IS_UNKNOWN(e, f_index)) {
          if (!e.typed_features.elementAt(f_index).getClass().getSimpleName().equals("Integer"))
            Dataset.perror("Feature.class :: Example feature " + f_index + " not an int");
          all_observed.addElement(
              new Integer(((Integer) e.typed_features.elementAt(f_index)).intValue()));
        }
      }

      all_observed_ordered = new int[all_observed.size()];
      for (i = 0; i < all_observed.size(); i++)
        all_observed_ordered[i] = all_observed.elementAt(i).intValue();
      Arrays.sort(all_observed_ordered);

      all_observed = new Vector<>();
      for (i = 0; i < all_observed_ordered.length; i++)
        if ((i == 0) || (all_observed_ordered[i] != all_observed_ordered[i - 1]))
          all_observed.addElement(new Integer(all_observed_ordered[i]));

      isplits_from_training = new int[all_observed.size() - 1];
      for (i = 0; i < isplits_from_training.length; i++)
        isplits_from_training[i] = all_observed.elementAt(i).intValue();
    }
  }

  Feature(String n, String t, Vector<String> m, double miv, double mav, boolean compute_tests) {
    formatted_for_a_dt = false;

    dsplits_from_training = null;
    isplits_from_training = null;
    dmin_index_in_dsplits_from_training =
        dmax_index_in_dsplits_from_training =
            imin_index_in_dsplits_from_training = imax_index_in_dsplits_from_training = -1;

    name = n;
    type = t;
    modalities = null;
    empty_domain = false;

    if (((Feature.IS_CONTINUOUS(t)) || (Feature.IS_INTEGER(t))) && (miv > mav))
      Dataset.perror(
          "Feature.class :: Continuous or Integer feature has min value "
              + miv
              + " > max value "
              + mav);
    else if ((Feature.IS_NOMINAL(t)) && (miv < mav))
      Dataset.perror(
          "Feature.class :: Nominal feature "
              + name
              + " has min value = "
              + miv
              + ", max value = "
              + mav
              + ", should be default Forbidden value "
              + Feature.FORBIDDEN_VALUE);

    if ((Feature.IS_CONTINUOUS(t)) && (miv >= mav))
      Dataset.perror(
          "Feature.class :: Continuous feature "
              + n
              + " has min value "
              + miv
              + " >= max value "
              + mav);

    if ((!Feature.IS_NOMINAL(t))
        && ((miv == (double) Feature.FORBIDDEN_VALUE) && (mav == (double) Feature.FORBIDDEN_VALUE)))
      Dataset.perror(
          "Feature.class :: Non nominal feature "
              + n
              + " has min value "
              + Feature.FORBIDDEN_VALUE
              + " == max value "
              + Feature.FORBIDDEN_VALUE
              + " = Forbidden value");

    if (Feature.IS_CONTINUOUS(t)) {
      dmin = miv;
      dmax = mav;

      dmin_from_data = dmax_from_data = (double) Feature.FORBIDDEN_VALUE;
      imin = imax = imin_from_data = imax_from_data = Feature.FORBIDDEN_VALUE;
    } else if (Feature.IS_INTEGER(t)) {
      imin = (int) miv;
      imax = (int) mav;

      imin_from_data = imax_from_data = Feature.FORBIDDEN_VALUE;
      dmin = dmax = dmin_from_data = dmax_from_data = (double) Feature.FORBIDDEN_VALUE;
    } else {
      imin = imax = imin_from_data = imax_from_data = Feature.FORBIDDEN_VALUE;
      dmin = dmax = dmin_from_data = dmax_from_data = (double) Feature.FORBIDDEN_VALUE;
    }

    if (Feature.IS_NOMINAL(t)) modalities = m;

    // generates tests for DTs
    if (compute_tests) tests = Feature.TEST_LIST(this);

    dispertion_statistic_value = -1.0;
  }

  // ALL PURPOSE INSTANCE METHODS

  public void try_format_tests(Dataset ds, int f_index, boolean compute_dmin_dmax) {
    if ((Discriminator_Tree.USE_OBSERVED_FEATURE_VALUES_FOR_SPLITS)
        && (Feature.IS_CONTINUOUS(type))) {
      if ((compute_dmin_dmax)
          && ((dmin_index_in_dsplits_from_training == -1)
              || (dmax_index_in_dsplits_from_training == -1)))
        compute_indexes_in_dsplits_from_training(ds, f_index);
      format_tests_using_observed_values_for_splits(ds, f_index);
    }
  }

  public void compute_indexes_in_dsplits_from_training(Dataset ds, int f_index) {
    double vmin = dmin;
    double vmax = dmax;
    double[] all_splits = ds.domain_feature(f_index).dsplits_from_training;

    boolean first_found = false;
    boolean last_found = false;
    int i;
    if (Feature.IS_CONTINUOUS(type)) {
      for (i = 0; i < all_splits.length; i++) {
        if ((all_splits[i] > vmin) && (all_splits[i] < vmax) && (!first_found)) {
          dmin_index_in_dsplits_from_training = i;
          first_found = true;
        } else if ((all_splits[i] >= vmax) && (!last_found)) {
          dmax_index_in_dsplits_from_training = i;
          last_found = true;
        }
      }
      if (!first_found) {
        System.out.print("\n Domain splits: ");
        for (i = 0; i < all_splits.length; i++) System.out.print(all_splits[i] + ", ");

        Dataset.perror(
            "Feature.class :: no split found for feature (dmin = "
                + dmin
                + ", dmax = "
                + dmax
                + ") "
                + toStringInTree(false, true));

      } else if (!last_found) dmax_index_in_dsplits_from_training = all_splits.length - 1;

    } else
      Dataset.perror(
          "Feature.class :: compute_indexes_in_dsplits_from_training sent for non-continuous"
              + " features");
  }

  public void format_tests_using_observed_values_for_splits(Dataset ds, int f_index) {
    formatted_for_a_dt = true;

    double vmin = dmin;
    double vmax = dmax;
    double[] all_splits = ds.domain_feature(f_index).dsplits_from_training;
    int i, j;
    if (Feature.IS_CONTINUOUS(type)) {
      Vector v = new Vector<Double>();
      if (dmax_index_in_dsplits_from_training >= dmin_index_in_dsplits_from_training)
        for (i = dmin_index_in_dsplits_from_training;
            i <= dmax_index_in_dsplits_from_training;
            i++) {
          if ((all_splits[i] <= vmin) || (all_splits[i] >= vmax)) {
            System.out.print("Domain splits\n");
            for (j = 0; j < all_splits.length; j++) System.out.print(all_splits[j] + ", ");

            Dataset.perror(
                "Feature.class :: vmin = "
                    + vmin
                    + ", all_splits["
                    + i
                    + "] = "
                    + all_splits[i]
                    + ", vmax = "
                    + vmax
                    + " for feature "
                    + toStringInTree(false, true));
          }
          v.addElement(new Double(all_splits[i]));
        }

      if (v.size() > 0) tests = v;
      else tests = null; // to enforce an Exception if used
    } else
      Dataset.perror(
          "Feature.class :: format_tests_using_observed_values_for_splits sent for non-continuous"
              + " features");
  }

  public boolean isSplittable() {
    if ((tests == null) || (tests.size() == 0)) return false;

    return true;
  }

  public boolean has_in_range(double v) {
    if ((Feature.IS_NOMINAL(type)) || (Feature.IS_INTEGER(type)))
      Dataset.perror("Feature.class :: feature " + this + " queried for double value " + v);
    if (!Feature.IS_CONTINUOUS(type))
      Dataset.perror("Feature.class :: feature type " + type + " unregistered ");
    if (v < dmin) return false;
    if (v > dmax) return false;
    return true;
  }

  public boolean has_in_range(int v) {
    if ((Feature.IS_NOMINAL(type)) || (Feature.IS_CONTINUOUS(type)))
      Dataset.perror("Feature.class :: feature " + this + " queried for double value " + v);
    if (!Feature.IS_INTEGER(type))
      Dataset.perror("Feature.class :: feature type " + type + " unregistered ");
    if (v < imin) return false;
    if (v > imax) return false;
    return true;
  }

  public boolean has_in_range(String s) {
    if ((Feature.IS_CONTINUOUS(type)) || (Feature.IS_INTEGER(type)))
      Dataset.perror(
          "Feature.class :: Continuous feature " + this + " queried for nominal value " + s);
    if (!Feature.IS_NOMINAL(type))
      Dataset.perror("Feature.class :: feature type " + type + " unregistered ");

    int i;
    String ss;
    for (i = 0; i < modalities.size(); i++) {
      ss = (String) modalities.elementAt(i);
      if (ss.equals(s)) return true;
    }
    return false;
  }

  public String range(boolean in_generator) {
    String v = "";
    int i;
    if (Feature.IS_NOMINAL(type)) {
      v += "{";
      for (i = 0; i < modalities.size(); i++) {
        v += "" + modalities.elementAt(i);
        if (i < modalities.size() - 1) v += ", ";
      }
      v += "}";
    } else if (Feature.IS_CONTINUOUS(type)) {
      if ((dmin_from_data != dmin) && (dmax_from_data != dmax) && (!in_generator)) v += "T: ";
      v += "[" + DF4.format(dmin) + ", " + DF4.format(dmax) + "]";
      if ((dmin_from_data != dmin)
          && (dmax_from_data != dmax)
          && ((dmin_from_data != dmax_from_data)
              || (dmin_from_data != (double) Feature.FORBIDDEN_VALUE))
          && (!in_generator))
        v += "; O: [" + DF4.format(dmin_from_data) + ", " + DF4.format(dmax_from_data) + "]";
    } else if (Feature.IS_INTEGER(type)) {
      if ((imin_from_data != imin) && (imax_from_data != imax) && (!in_generator)) v += "T: ";

      if (imax == imin) v += "{" + imin + "}";
      else {
        v += "{" + imin + ", " + (imin + 1);
        if (imax > imin + 2) v += ", ...";
        if (imax > imin + 1) v += ", " + imax;
        v += "}";
      }
      if ((imin_from_data != imin)
          && (imax_from_data != imax)
          && ((imin_from_data != imax_from_data) || (imin_from_data != Feature.FORBIDDEN_VALUE))
          && (!in_generator)) {
        v += "; O: {" + imin_from_data + ", " + (imin_from_data + 1);
        if (imax_from_data > imin_from_data + 1) v += ", ...";
        v += ", " + imax_from_data + "}";
      }
    }
    return v;
  }

  public String toString() {
    String v = "";
    int i;
    v +=
        name
            + " -- "
            + type
            + " in "
            + range(false)
            + " ["
            + Feature.DISPERSION_NAME[Feature.TYPE_INDEX(type)]
            + " = "
            + DF4.format(dispertion_statistic_value)
            + "]";

    if (Feature.DISPLAY_TESTS) v += " -- tests[" + tests.size() + "] : " + tests(false);

    return v;
  }

  public String toStringInTree(boolean internalnode, boolean display_tests) {
    String v = "";
    int i;
    if (internalnode) v += name + " (" + type + ") in " + range(true);
    else v += "(" + name + " in " + range(true) + ")";

    if (display_tests) v += " -- tests : " + tests(display_tests);

    if ((internalnode) || (display_tests)) v += ";";

    return v;
  }

  // DISCRIMINATOR METHODS

  public String display_test(int index_test) {
    String v = name;
    int i;
    Vector ssv;

    if (Feature.IS_CONTINUOUS(type))
      v += " <= " + DF6.format(((Double) tests.elementAt(index_test)).doubleValue());
    else if (Feature.IS_INTEGER(type))
      v += " <= " + ((Integer) tests.elementAt(index_test)).intValue();
    else if (Feature.IS_NOMINAL(type)) {
      v += " in {";
      ssv = (Vector) tests.elementAt(index_test);
      for (i = 0; i < ssv.size(); i++) {
        v += (String) ssv.elementAt(i) + " ";
        if (i < ssv.size() - 1) v += ", ";
      }
      v += "}";
    } else Dataset.perror("Feature.class :: no type available for feature " + this);
    return v;
  }

  public boolean example_goes_left(
      Example e,
      int index_feature_in_e,
      int index_test,
      boolean unspecified_attribute_handling_biased) {
    // path followed in the tree by an example
    // continuous OR integer values : <= is left, > is right
    // nominal values : in the set is left, otherwise is right

    // unspecified_attribute_handling_biased = true => uses local domain and split to decide random
    // branching, else Bernoulli(0.5)

    double cv, tv;
    int ci, ti;
    String nv;
    Vector ssv;
    int i;

    double p_left;

    // New: takes into account unknown feature values
    if (Example.FEATURE_IS_UNKNOWN(e, index_feature_in_e)) {
      if (!unspecified_attribute_handling_biased) {
        if (Algorithm.RANDOM_P_NOT_HALF() < 0.5) return true;
        else return false;
      } else {
        p_left = -1.0;
        if (Feature.IS_CONTINUOUS(type)) {
          if (dmax == dmin) Dataset.perror("Feature.class :: dmax = " + dmax + " == dmin ");
          if (((Double) tests.elementAt(index_test)).doubleValue() < dmin)
            Dataset.perror(
                "Feature.class :: test = "
                    + ((Double) tests.elementAt(index_test)).doubleValue()
                    + " < dmin  = "
                    + dmin);
          if (((Double) tests.elementAt(index_test)).doubleValue() > dmax)
            Dataset.perror(
                "Feature.class :: test = "
                    + ((Double) tests.elementAt(index_test)).doubleValue()
                    + " > dmax  = "
                    + dmax);

          p_left = (((Double) tests.elementAt(index_test)).doubleValue() - dmin) / (dmax - dmin);
        } else if (Feature.IS_INTEGER(type)) {
          if (imax == imin) Dataset.perror("Feature.class :: imax = " + imax + " == imin ");
          if (((Integer) tests.elementAt(index_test)).intValue() < imin)
            Dataset.perror(
                "Feature.class :: test = "
                    + ((Integer) tests.elementAt(index_test)).intValue()
                    + " < imin  = "
                    + imin);
          if (((Integer) tests.elementAt(index_test)).intValue() > imax)
            Dataset.perror(
                "Feature.class :: test = "
                    + ((Integer) tests.elementAt(index_test)).intValue()
                    + " > imax  = "
                    + imax);

          p_left =
              ((double) (((Integer) tests.elementAt(index_test)).intValue() - imin + 1))
                  / ((double) imax - imin + 1);
        } else if (Feature.IS_NOMINAL(type))
          p_left = (((Vector) tests.elementAt(index_test)).size()) / ((double) modalities.size());
        else Dataset.perror("Feature.class :: no type available for feature " + this);
        if (Algorithm.RANDOM_P_NOT(p_left) < p_left) return true;
        else return false;
      }
    }

    if (Feature.IS_CONTINUOUS(type)) {
      if ((e.typed_features
              .elementAt(index_feature_in_e)
              .getClass()
              .getSimpleName()
              .equals("String"))
          || (e.typed_features
              .elementAt(index_feature_in_e)
              .getClass()
              .getSimpleName()
              .equals("Integer")))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a Double");

      cv = ((Double) e.typed_features.elementAt(index_feature_in_e)).doubleValue();
      tv = ((Double) tests.elementAt(index_test)).doubleValue();
      if (cv <= tv) return true;
      return false;
    } else if (Feature.IS_INTEGER(type)) {
      if ((e.typed_features
              .elementAt(index_feature_in_e)
              .getClass()
              .getSimpleName()
              .equals("String"))
          || (e.typed_features
              .elementAt(index_feature_in_e)
              .getClass()
              .getSimpleName()
              .equals("Double")))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a Double");

      ci = ((Integer) e.typed_features.elementAt(index_feature_in_e)).intValue();
      ti = ((Integer) tests.elementAt(index_test)).intValue();
      if (ci <= ti) return true;
      return false;
    } else if (Feature.IS_NOMINAL(type)) {
      if ((e.typed_features
              .elementAt(index_feature_in_e)
              .getClass()
              .getSimpleName()
              .equals("Double"))
          || (e.typed_features
              .elementAt(index_feature_in_e)
              .getClass()
              .getSimpleName()
              .equals("Integer")))
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
}
