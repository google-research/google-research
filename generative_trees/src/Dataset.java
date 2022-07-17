// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.*;

public class Dataset implements Debuggable {
  public static String KEY_NODES = "@NODES";
  public static String KEY_ARCS = "@ARCS";

  public static String PATH_GENERATIVE_MODELS = "generators";

  public static String KEY_SEPARATION_STRING[] = {"\t", ","};
  public static int SEPARATION_INDEX = 1;

  public static String DEFAULT_DIR = "Datasets", SEP = "/", KEY_COMMENT = "//";

  public static String START_KEYWORD = "@";

  public static String SUFFIX_FEATURES = "features";
  public static String SUFFIX_EXAMPLES = "data";

  public static int NUMBER_GENERATED_EXAMPLES_DEFAULT = 100;
  public static int NUMBER_GENERATED_EXAMPLES_DEFAULT_HISTOGRAMS = 5000;

  int number_features_total, number_examples_total_from_file, number_examples_total_generated;

  private Vector<Feature> features;
  // putting them private to prevent mistakes on computing splits in GTs / DTs by a direct access

  Vector<Histogram> domain_histograms;
  Vector<Example> examples_from_file;
  // examples_from_file is the stored / read data

  Vector<Example> examples_generated;
  // additional sets of examples generated

  Domain myDomain;
  String[] features_names_from_file;

  public static void perror(String error_text) {
    System.out.println("\n" + error_text);
    System.out.println("\nExiting to system\n");
    System.exit(1);
  }

  public static void warning(String warning_text) {
    System.out.print(" * WARNING * " + warning_text);
  }

  Dataset(String dir, String pref, Domain d) {
    myDomain = d;

    examples_generated = null;
    domain_histograms = null;

    number_examples_total_generated = Dataset.NUMBER_GENERATED_EXAMPLES_DEFAULT;
    features_names_from_file = null;
  }

  Dataset(Domain d) {
    myDomain = d;
    number_examples_total_generated = -1;
    features_names_from_file = null;
  }

  public Feature domain_feature(int f_index) {
    return features.elementAt(f_index);
  }

  public int number_domain_features() {
    return features.size();
  }

  public void printFeatures() {
    int i;
    System.out.println(features.size() + " features : ");
    for (i = 0; i < features.size(); i++) System.out.println((Feature) features.elementAt(i));
  }

  public void load_features_and_examples() {
    if (myDomain.myW == null) Dataset.perror("Dataset.class :: use not authorised without Wrapper");

    FileReader e;
    BufferedReader br;
    StringTokenizer t;

    Vector<Vector<String>> examples_read = new Vector<>();
    Vector<String> current_example;
    String[] features_types = null;

    String dum, n, ty;
    int i, j;

    boolean feature_names_ok = false;

    // features
    features = new Vector<>();

    System.out.print(
        "\nLoading features & examples data... " + myDomain.myW.path_and_name_of_domain_dataset);

    number_features_total = -1;
    try {
      e = new FileReader(myDomain.myW.path_and_name_of_domain_dataset);
      br = new BufferedReader(e);

      while ((dum = br.readLine()) != null) {
        if ((dum.length() > 1) && (!dum.substring(0, KEY_COMMENT.length()).equals(KEY_COMMENT))) {
          if (!feature_names_ok) {
            // the first usable line must be feature names
            t = new StringTokenizer(dum, KEY_SEPARATION_STRING[SEPARATION_INDEX]);
            features_names_from_file = new String[t.countTokens()];

            i = 0;
            while (t.hasMoreTokens()) {
              n = t.nextToken();
              // must be String
              if (is_number(n))
                Dataset.perror(
                    "Dataset.class :: String "
                        + n
                        + " identified as Number; should not be the case for feature names");
              features_names_from_file[i] = n;
              i++;
            }

            number_features_total = features_names_from_file.length;
            feature_names_ok = true;
          } else {
            // records all following values; checks sizes comply
            current_example = new Vector<>();
            t = new StringTokenizer(dum, KEY_SEPARATION_STRING[SEPARATION_INDEX]);
            if (t.countTokens() != number_features_total)
              Dataset.perror(
                  "Dataset.class :: Example string + "
                      + dum
                      + " does not have "
                      + number_features_total
                      + " features");
            while (t.hasMoreTokens()) current_example.addElement(t.nextToken());
            examples_read.addElement(current_example);
          }
        }
      }
    } catch (IOException eee) {
      System.out.println(
          "Problem loading "
              + myDomain.myW.path_and_name_of_domain_dataset
              + " file --- Check the access to file");
      System.exit(0);
    }

    // checking, if density plot requested, that names are found in columns and computes
    // myW.index_x_name , index_y_name
    if ((myDomain.myW.x_name != null) && ((myDomain.myW.y_name != null))) {
      myDomain.myW.index_x_name = myDomain.myW.index_y_name = -1;
      for (i = 0; i < features_names_from_file.length; i++)
        if (features_names_from_file[i].equals(myDomain.myW.x_name)) myDomain.myW.index_x_name = i;
        else if (features_names_from_file[i].equals(myDomain.myW.y_name))
          myDomain.myW.index_y_name = i;
      if (myDomain.myW.index_x_name == -1)
        Dataset.perror(
            "Dataset.class :: variable density name "
                + myDomain.myW.x_name
                + " not found in dataset's variable names");
      if (myDomain.myW.index_y_name == -1)
        Dataset.perror(
            "Dataset.class :: variable density name "
                + myDomain.myW.y_name
                + " not found in dataset's variable names");
    }

    System.out.print(
        "ok (#"
            + number_features_total
            + " features total, #"
            + examples_read.size()
            + " examples total)... Computing features... ");
    features_types = new String[number_features_total];
    boolean not_a_number, not_an_integer, not_a_binary;
    int idum, nvalf;
    double ddum;
    double miv, mav;
    Vector<String> v;

    // compute types
    for (i = 0; i < number_features_total; i++) {

      // String = nominal ? 1: text
      not_a_number = false;
      j = 0;
      do {
        n = examples_read.elementAt(j).elementAt(i);
        if (!n.equals(Unknown_Feature_Value.S_UNKNOWN)) {
          if (!is_number(n)) {
            not_a_number = true;
          } else j++;
        } else j++;
      } while ((!not_a_number) && (j < examples_read.size()));

      // String = nominal ? 2: binary
      if ((!not_a_number) && (myDomain.myW.force_binary_coding)) {
        j = 0;
        not_a_binary = false;
        do {
          n = examples_read.elementAt(j).elementAt(i);
          if (!n.equals(Unknown_Feature_Value.S_UNKNOWN)) {
            not_an_integer = false;
            idum = -1;
            try {
              idum = Integer.parseInt(n);
            } catch (NumberFormatException nfe) {
              not_an_integer = true;
            }

            if ((not_an_integer) || ((idum != 0) && (idum != 1))) {
              not_a_binary = true;
            } else j++;
          } else j++;
        } while ((!not_a_binary) && (j < examples_read.size()));
        if (!not_a_binary) not_a_number = true;
      }

      if (not_a_number) features_types[i] = Feature.NOMINAL;
      else if (!myDomain.myW.force_integer_coding) features_types[i] = Feature.CONTINUOUS;
      else {
        // makes distinction integer / real
        not_an_integer = false;
        j = 0;
        do {
          n = examples_read.elementAt(j).elementAt(i);
          if (!n.equals(Unknown_Feature_Value.S_UNKNOWN)) {
            try {
              idum = Integer.parseInt(n);
            } catch (NumberFormatException nfe) {
              not_an_integer = true;
            }
            if (!not_an_integer) j++;
          } else j++;
        } while ((!not_an_integer) && (j < examples_read.size()));
        if (not_an_integer) features_types[i] = Feature.CONTINUOUS;
        else features_types[i] = Feature.INTEGER;
      }
    }

    System.out.print("Types found: [");
    for (i = 0; i < number_features_total; i++) {
      System.out.print(features_types[i]);
      if (i < number_features_total - 1) System.out.print(",");
    }
    System.out.print("] ");

    // compute features
    boolean value_seen;
    for (i = 0; i < number_features_total; i++) {
      value_seen = false;
      miv = mav = -1.0;
      v = null;
      if (Feature.IS_NOMINAL(features_types[i])) {
        v = new Vector();
        for (j = 0; j < examples_read.size(); j++)
          if ((!examples_read.elementAt(j).elementAt(i).equals(Unknown_Feature_Value.S_UNKNOWN))
              && (!v.contains(examples_read.elementAt(j).elementAt(i))))
            v.addElement(examples_read.elementAt(j).elementAt(i));

        features.addElement(
            new Feature(features_names_from_file[i], features_types[i], v, miv, mav, true));
      } else {
        nvalf = 0;
        for (j = 0; j < examples_read.size(); j++) {
          n = examples_read.elementAt(j).elementAt(i);
          if (!n.equals(Unknown_Feature_Value.S_UNKNOWN)) {
            nvalf++;
            ddum = Double.parseDouble(n);
            if (!value_seen) {
              miv = mav = ddum;
              value_seen = true;
            } else {
              if (ddum < miv) miv = ddum;
              if (ddum > mav) mav = ddum;
            }
          }
        }
        if (nvalf == 0)
          Dataset.perror(
              "Dataset.class :: feature "
                  + features_names_from_file[i]
                  + " has only unknown values");
        features.addElement(
            new Feature(features_names_from_file[i], features_types[i], v, miv, mav, true));
      }
    }

    System.out.print("ok... Computing examples... ");
    examples_from_file = new Vector<>();

    Example ee;
    number_examples_total_from_file = examples_read.size();
    Dataset.NUMBER_GENERATED_EXAMPLES_DEFAULT = number_examples_total_from_file;
    number_examples_total_generated = Dataset.NUMBER_GENERATED_EXAMPLES_DEFAULT;

    for (j = 0; j < examples_read.size(); j++) {
      ee = new Example(j, examples_read.elementAt(j), features, true);
      examples_from_file.addElement(ee);
      if (ee.contains_unknown_values()) myDomain.myW.has_missing_values = true;
    }

    int errfound = 0;
    for (i = 0; i < number_examples_total_from_file; i++) {
      ee = (Example) examples_from_file.elementAt(i);
      errfound += ee.checkAndCompleteFeatures(features);
    }
    if (errfound > 0)
      Dataset.perror(
          "Dataset.class :: found "
              + errfound
              + " errs for feature domains in examples. Please correct domains in .features file."
              + " ");

    compute_feature_statistics();

    System.out.println("ok... ");
  }

  public boolean is_number(String n) {
    double test;
    boolean is_double = true;
    try {
      test = Double.parseDouble(n);
    } catch (NumberFormatException nfe) {
      is_double = false;
    }
    if (is_double) return true;

    ParsePosition p = new ParsePosition(0);
    NumberFormat.getNumberInstance().parse(n, p);
    return (n.length() == p.getIndex());
  }

  public void compute_feature_statistics() {
    int i, j;
    double[] probab = null;
    double expect_X2 = 0.0, expect_squared = 0.0, vex = -1.0, tot;
    double nfeat = 0.0, nnonz = 0.0;
    for (i = 0; i < features.size(); i++) {
      nfeat = 0.0;
      if (Feature.IS_NOMINAL(((Feature) features.elementAt(i)).type))
        probab = new double[((Feature) features.elementAt(i)).modalities.size()];
      else {
        expect_X2 = expect_squared = 0.0;
      }

      for (j = 0; j < number_examples_total_from_file; j++)
        if (!Example.FEATURE_IS_UNKNOWN((Example) examples_from_file.elementAt(j), i))
          if (Feature.IS_NOMINAL(((Feature) features.elementAt(i)).type))
            probab[
                    ((Feature) features.elementAt(i))
                        .modalities.indexOf(
                            (String)
                                ((Example) examples_from_file.elementAt(j))
                                    .typed_features.elementAt(i))] +=
                1.0;
          else {
            if (Feature.IS_INTEGER(((Feature) features.elementAt(i)).type))
              vex =
                  (double)
                      ((Integer)
                              ((Example) examples_from_file.elementAt(j))
                                  .typed_features.elementAt(i))
                          .intValue();
            else if (Feature.IS_CONTINUOUS(((Feature) features.elementAt(i)).type))
              vex =
                  ((Double) ((Example) examples_from_file.elementAt(j)).typed_features.elementAt(i))
                      .doubleValue();
            else
              Dataset.perror(
                  "Dataset.class :: no feature type " + ((Feature) features.elementAt(i)).type);

            expect_squared += vex;
            expect_X2 += (vex * vex);
            nnonz += 1.0;
          }

      if (Feature.IS_NOMINAL(((Feature) features.elementAt(i)).type)) {
        tot = 0.0;
        for (j = 0; j < probab.length; j++) tot += probab[j];
        for (j = 0; j < probab.length; j++) probab[j] /= tot;
        ((Feature) features.elementAt(i)).dispertion_statistic_value =
            Statistics.SHANNON_ENTROPY(probab);
      } else {
        if (nnonz == 0.0)
          Dataset.perror("Dataset.class :: feature " + i + " has only unknown values");

        expect_X2 /= nnonz;
        expect_squared /= nnonz;
        expect_squared *= expect_squared;
        ((Feature) features.elementAt(i)).dispertion_statistic_value = expect_X2 - expect_squared;
      }
    }
  }

  public void compute_domain_splits(int cv_fold) {
    int i;
    for (i = 0; i < features.size(); i++)
      if ((Feature.IS_INTEGER(((Feature) features.elementAt(i)).type))
          || (Feature.IS_CONTINUOUS(((Feature) features.elementAt(i)).type)))
        ((Feature) features.elementAt(i)).init_splits(this, i, cv_fold);
  }

  public void compute_domain_histograms() {
    domain_histograms = new Vector<>();
    Histogram h;
    int i;
    boolean ok;
    for (i = 0; i < features.size(); i++) {
      h = new Histogram(i, features.elementAt(i));
      ok = h.fill_histogram(examples_from_file, true);
      if (!ok)
        Dataset.perror(
            "Dataset.class :: not a single domain example with non UNKNOWN feature value for"
                + " feature #"
                + i);
      domain_histograms.addElement(h);
    }
  }

  public Histogram from_histogram(int feature_index) {
    Histogram h = new Histogram(feature_index, features.elementAt(feature_index));
    // replaces bucket labels by the feature's
    int i;
    for (i = 0; i < (domain_histograms.elementAt(feature_index)).histogram_features.size(); i++) {
      h.histogram_features.setElementAt(
          Feature.copyOf(
              (domain_histograms.elementAt(feature_index)).histogram_features.elementAt(i), false),
          i);
    }
    return h;
  }

  public void generate_examples(Generator_Tree gt) {
    double total = 0.0;
    int i;
    examples_generated = gt.generate_sample(number_examples_total_generated);
    compute_sample_normalized_weight(false, -1);
  }

  public void generate_and_replace_examples(Generator_Tree gt, Generator_Node gn) {
    // gn = leaf split in GT

    double total = 0.0;
    int i;

    gt.replace_sample(examples_generated, gn);
    compute_sample_normalized_weight(false, -1);
  }

  public void compute_sample_normalized_weight(boolean positive_real, int fold) {
    if ((positive_real) || (fold != -1))
      Dataset.perror("Dataset.class :: bad parameters for compute_sample_normalized_weight");

    double total = 0.0;
    int i;
    for (i = 0; i < examples_generated.size(); i++)
      total += ((Example) examples_generated.elementAt(i)).unnormalized_weight;
    for (i = 0; i < examples_generated.size(); i++)
      ((Example) examples_generated.elementAt(i)).compute_sample_normalized_weight(total);
  }

  public String toString() {
    int i, nm = examples_from_file.size();
    if (examples_from_file.size() > 10) nm = 10;

    String v = "\n* Features (Unknown value coded as " + Unknown_Feature_Value.S_UNKNOWN + ") --\n";
    for (i = 0; i < features.size(); i++) v += ((Feature) features.elementAt(i)).toString() + "\n";
    v += "\n* First examples --\n";
    for (i = 0; i < nm; i++) v += ((Example) examples_from_file.elementAt(i)).toString();

    if (myDomain.myW == null) { // did not use a wrapper
      v += "\n* Domain histograms --\n";
      for (i = 0; i < domain_histograms.size(); i++) {
        v += (domain_histograms.elementAt(i)).toStringSave();
        v += "//\n";
      }
    }

    return v;
  }

  public void printFirstGeneratedExamples() {
    int i, nm = examples_generated.size();
    if (examples_generated.size() > 10) nm = 10;
    for (i = 0; i < nm; i++) System.out.print((Example) examples_generated.elementAt(i));
    System.out.println("");
  }

  // methods related to returning examples

  public void sample_check() {
    if (examples_generated == null)
      Dataset.perror("Dataset.class :: no generated sample available");
  }

  // sizes

  private int generated_train_size() { // -- CHECK no direct call to this but train_size instead
    sample_check();
    return examples_generated.size();
  }

  private int real_train_size(int fold) {
    if (fold != -1) Dataset.perror("Dataset.class :: should not use fold = " + fold);

    return examples_from_file.size();
  }

  public int train_size(int fold, boolean use_real) {
    if (fold != -1) Dataset.perror("Dataset.class :: should not use fold = " + fold);

    if (use_real) return real_train_size(fold);
    else return generated_train_size();
  }

  // examples

  private Example generated_train_example(
      int nex) { // train_example_generated -- CHECK no direct call to this but train_example
    // instead
    return (Example) examples_generated.elementAt(nex);
  }

  private Example real_train_example(
      int fold,
      int nex) { // train_example_from_file -- CHECK no direct call to this but train_example
    // instead
    if (fold != -1) Dataset.perror("Dataset.class :: should not use fold = " + fold);

    return (Example) examples_from_file.elementAt(nex);
  }

  public Example train_example(int fold, int nex, boolean use_real) {
    if (fold != -1) Dataset.perror("Dataset.class :: should not use fold = " + fold);

    if (use_real) return real_train_example(fold, nex);
    else return generated_train_example(nex);
  }
}
