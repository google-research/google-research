import java.io.*;
import java.util.*;

public class Dataset implements Debuggable {

  public static String KEY_SEPARATION_TYPES[] = {"@TABULATION", "@COMMA"};
  public static String KEY_SEPARATION_STRING[] = {"\t", ","};
  public static int SEPARATION_INDEX = 1;

  public static String DEFAULT_DIR = "Datasets", SEP = "/", KEY_COMMENT = "//";

  public static String KEY_DIRECTORY = "@DIRECTORY";
  public static String KEY_PREFIX = "@PREFIX";
  public static String KEY_ALGORITHM = "@ALGORITHM";
  public static String KEY_NCT = "@NCT";
  public static String KEY_NOISE = "@ETA_NOISE";
  public static String KEY_CHECK_STRATIFIED_LABELS = "@CHECK_STRATIFIED_LABELS";

  public static String START_KEYWORD = "@";
  public static String FIT_CLASS = "@FIT_CLASS";
  public static String[] FIT_CLASS_MODALITIES = {"NONE", "MEAN", "MEDIAN", "MINMAX", "BINARY"};
  // NONE means the problem has two class modalities -> {-1, 1}, default value
  // Otherwise, classes are fit in {-1,1} the keyword gives the meaning of the 0 value for the class
  // :
  // * MEAN -> classes are translated so that mean = 0 and then shrunk to fit in [-1,1]
  // * MEDIAN -> classes are translated so that median = 0 and then shrunk to fit in [-1,1]
  // * MINMAX -> classes are translated so that (min+max)/2 = 0 and then shrunk to fit in [-1,1]
  // * BINARY -> classes are translated so that mean = 0 and then BINARIZED in {-1, +1} with signs

  public static int DEFAULT_INDEX_FIT_CLASS = 0;

  public static int INDEX_FIT_CLASS(String s) {
    int i;
    for (i = 0; i < FIT_CLASS_MODALITIES.length; i++)
      if (s.equals(FIT_CLASS_MODALITIES[i])) return i;
    Dataset.perror("Value " + s + " for keyword " + FIT_CLASS + " not recognized");
    return -1;
  }

  public static double TRANSLATE_SHRINK(
      double v, double translate_v, double min_v, double max_v, double max_m) {
    if ((v < min_v) || (v > max_v))
      Dataset.perror("Value " + v + " is not in the interval [" + min_v + ", " + max_v + "]");

    if (max_v < min_v) Dataset.perror("Max " + max_v + " < Min " + min_v);

    if (max_v == min_v) return v;

    double delta = (v - translate_v);
    double mm;
    if (Math.abs(max_v) > Math.abs(min_v)) mm = Math.abs(max_v - translate_v);
    else mm = Math.abs(min_v - translate_v);

    return ((delta / mm) * max_m);
  }

  public static double CENTER(double v, double avg, double stddev) {
    if (stddev == 0.0) {
      Dataset.warning("Standard deviation is zero");
      return 0.0;
    }
    return ((v - avg) / stddev);
  }

  public static String SUFFIX_FEATURES = "features";
  public static String SUFFIX_EXAMPLES = "data";

  int index_class, number_initial_features, number_real_features, number_examples_total;

  double eta_noise;
  // training noise in the LS model

  String nameFeatures, nameExamples, pathSave, domainName;
  Vector features, examples;
  // WARNING : features also includes class
  int[] index_observation_features_to_index_features;

  Vector stratified_sample, training_sample, test_sample;

  Domain myDS;

  public static void perror(String error_text) {
    System.out.println("\n" + error_text);
    System.out.println("\nExiting to system\n");
    System.exit(1);
  }

  public static void warning(String warning_text) {
    System.out.println(" * WARNING * " + warning_text);
  }

  Dataset(String dir, String pref, Domain d, double eta) {
    myDS = d;

    eta_noise = eta;

    domainName = pref;

    nameFeatures = dir + SEP + pref + SEP + pref + "." + SUFFIX_FEATURES;
    nameExamples = dir + SEP + pref + SEP + pref + "." + SUFFIX_EXAMPLES;
    pathSave = dir + SEP + pref + SEP;
  }

  public void printFeatures() {
    int i;
    System.out.println(features.size() + " features : ");
    for (i = 0; i < features.size(); i++) System.out.println((Feature) features.elementAt(i));
    System.out.println("Class index : " + index_class);
  }

  public void load_features() {
    FileReader e;
    BufferedReader br;
    StringTokenizer t;
    String dum, n, ty;
    Vector v = null;
    Vector dumv = new Vector();

    int i, index = 0;

    features = new Vector();
    index_class = -1;

    try {
      e = new FileReader(nameFeatures);
      br = new BufferedReader(e);

      while ((dum = br.readLine()) != null) {
        if ((dum.length() == 1)
            || ((dum.length() > 1)
                && (!dum.substring(0, KEY_COMMENT.length()).equals(KEY_COMMENT)))) {
          if (dum.substring(0, 1).equals(Dataset.START_KEYWORD)) {
            t = new StringTokenizer(dum, KEY_SEPARATION_STRING[SEPARATION_INDEX]);
            if (t.countTokens() < 2) Dataset.perror("No value for keyword " + t.nextToken());
            n = t.nextToken();
            if (n.equals(Dataset.FIT_CLASS))
              Dataset.DEFAULT_INDEX_FIT_CLASS = Dataset.INDEX_FIT_CLASS(t.nextToken());
          } else {
            t = new StringTokenizer(dum, KEY_SEPARATION_STRING[SEPARATION_INDEX]);
            if (t.countTokens() > 0) {
              n = t.nextToken();
              ty = t.nextToken();

              if (Feature.INDEX(ty) == Feature.CLASS_INDEX) {
                if (index_class != -1)
                  Dataset.perror("At least two classes named such in feature file");
                else index_class = index;
              } else {
                dumv.addElement(new Integer(index));
              }

              if (Feature.HAS_MODALITIES(ty)) {
                v = new Vector();
                while (t.hasMoreTokens()) v.addElement(t.nextToken());
              }

              features.addElement(new Feature(n, ty, v));
              index++;
            }
          }
        }
      }
      e.close();
    } catch (IOException eee) {
      System.out.println(
          "Problem loading ."
              + SUFFIX_FEATURES
              + " file --- Check the access to file "
              + nameFeatures
              + "...");
      System.exit(0);
    }

    index_observation_features_to_index_features = new int[dumv.size()];
    for (i = 0; i < dumv.size(); i++)
      index_observation_features_to_index_features[i] = ((Integer) dumv.elementAt(i)).intValue();

    Dataset.warning(
        "Class renormalization using method : "
            + Dataset.FIT_CLASS_MODALITIES[DEFAULT_INDEX_FIT_CLASS]);

    number_initial_features = features.size() - 1;
    if (Debug) System.out.println("Found " + features.size() + " features, including class");
  }

  public void load_examples() {
    FileReader e;
    BufferedReader br;
    StringTokenizer t;
    String dum;
    Vector v = null;
    Double dd;

    examples = new Vector();
    Example ee = null;
    int i, j, idd = 0, nex = 0;

    Vector<Vector<Double>> continuous_features_values = new Vector<>();
    for (i = 0; i < features.size(); i++)
      if ((i != index_class) && (Feature.IS_CONTINUOUS(((Feature) features.elementAt(i)).type)))
        continuous_features_values.addElement(new Vector<Double>());
      else continuous_features_values.addElement(null);

    // Computing the whole number of examples

    try {
      e = new FileReader(nameExamples);
      br = new BufferedReader(e);

      while ((dum = br.readLine()) != null) {
        if ((dum.length() == 1)
            || ((dum.length() > 1)
                && (!dum.substring(0, KEY_COMMENT.length()).equals(KEY_COMMENT)))) {
          t = new StringTokenizer(dum, KEY_SEPARATION_STRING[SEPARATION_INDEX]);
          if (t.countTokens() > 0) {
            nex++;
          }
        }
      }
      e.close();
    } catch (IOException eee) {
      System.out.println(
          "Problem loading ."
              + SUFFIX_FEATURES
              + " file --- Check the access to file "
              + nameFeatures
              + "...");
      System.exit(0);
    }

    if (SAVE_MEMORY) System.out.print(nex + " examples to load... ");

    number_examples_total = 0;
    try {
      e = new FileReader(nameExamples);
      br = new BufferedReader(e);

      while ((dum = br.readLine()) != null) {
        if ((dum.length() == 1)
            || ((dum.length() > 1)
                && (!dum.substring(0, KEY_COMMENT.length()).equals(KEY_COMMENT)))) {
          t = new StringTokenizer(dum, KEY_SEPARATION_STRING[SEPARATION_INDEX]);
          if (t.countTokens() > 0) {
            v = new Vector();
            while (t.hasMoreTokens())
              v.addElement(t.nextToken()); // v contains the class information
            ee = new Example(idd, v, index_class, features);
            number_examples_total++;

            examples.addElement(ee);

            number_real_features = ee.typed_features.size();
            idd++;

            for (i = 0; i < features.size(); i++)
              if ((i != index_class)
                  && (Feature.IS_CONTINUOUS(((Feature) features.elementAt(i)).type))) {
                dd = new Double(((Double) ee.typed_features.elementAt(i)).doubleValue());
                if (!continuous_features_values.elementAt(i).contains(dd))
                  continuous_features_values.elementAt(i).addElement(dd);
              }

            if (SAVE_MEMORY)
              if (idd % (nex / 20) == 0)
                System.out.print(((idd / (nex / 20)) * 5) + "% " + myDS.memString() + " ");
          }
        }
      }
      e.close();
    } catch (IOException eee) {
      System.out.println(
          "Problem loading ."
              + SUFFIX_EXAMPLES
              + " file --- Check the access to file "
              + nameExamples
              + "...");
      System.exit(0);
    }

    if (number_examples_total != nex)
      Dataset.perror("Dataset.class :: mismatch in the number of examples");

    if (SAVE_MEMORY) System.out.print("ok. \n");

    double[] all_vals;
    for (i = 0; i < features.size(); i++)
      if ((i != index_class) && (Feature.IS_CONTINUOUS(((Feature) features.elementAt(i)).type))) {
        all_vals = new double[continuous_features_values.elementAt(i).size()];
        for (j = 0; j < continuous_features_values.elementAt(i).size(); j++)
          all_vals[j] = continuous_features_values.elementAt(i).elementAt(j).doubleValue();
        QuickSort.quicksort(all_vals);
        ((Feature) features.elementAt(i)).update_tests(all_vals);
      }

    // normalizing classes

    double[] all_classes = new double[number_examples_total];
    for (i = 0; i < number_examples_total; i++)
      all_classes[i] = ((Example) examples.elementAt(i)).unnormalized_class;
    double min_c, max_c, tv;

    max_c = min_c = tv = 0.0;
    for (i = 0; i < number_examples_total; i++) {
      if ((i == 0) || (max_c < all_classes[i])) max_c = all_classes[i];
      if ((i == 0) || (min_c > all_classes[i])) min_c = all_classes[i];
    }

    if (DEFAULT_INDEX_FIT_CLASS == 0) {
      // checks that there is only two modalities
      for (i = 0; i < number_examples_total; i++)
        if ((all_classes[i] != min_c) && (all_classes[i] != max_c))
          Dataset.perror(
              "class value " + all_classes[i] + " should be either " + min_c + " or " + max_c);
      tv = (min_c + max_c) / 2.0;
    } else if ((DEFAULT_INDEX_FIT_CLASS == 1) || (DEFAULT_INDEX_FIT_CLASS == 4)) {
      tv = 0.0;
      for (i = 0; i < number_examples_total; i++) tv += all_classes[i];
      tv /= (double) number_examples_total;
    } else if (DEFAULT_INDEX_FIT_CLASS == 2) {
      tv = 0.0;
      QuickSort.quicksort(all_classes);
      tv = all_classes[number_examples_total / 2];
    } else if (DEFAULT_INDEX_FIT_CLASS == 3) {
      tv = (min_c + max_c) / 2.0;
    }

    // end

    int errfound = 0;
    for (i = 0; i < number_examples_total; i++) {
      ee = (Example) examples.elementAt(i);
      ee.complete_normalized_class(tv, min_c, max_c, eta_noise);

      errfound += ee.checkFeatures(features, index_class);
    }
    if (errfound > 0)
      Dataset.perror(
          "Dataset.class :: found "
              + errfound
              + " errs for feature domains in examples. Please correct domains in .features file."
              + " ");

    if (Debug) System.out.println("Found " + examples.size() + " examples");
  }

  public double getProportionExamplesSign(boolean positive) {
    int i;
    double cc, tot = 0.0;
    for (i = 0; i < number_examples_total; i++) {
      cc = ((Example) examples.elementAt(i)).normalized_class;
      if ((positive) && (cc >= 0.0)) tot++;
      else if ((!positive) && (cc < 0.0)) tot++;
    }
    tot /= (double) number_examples_total;
    return tot;
  }

  public double getProportionExamplesSign(Vector v, boolean positive) {
    int i;
    double cc, tot = 0.0;
    for (i = 0; i < v.size(); i++) {
      cc = ((Example) examples.elementAt(((Integer) v.elementAt(i)).intValue())).normalized_class;
      if ((positive) && (cc >= 0.0)) tot++;
      else if ((!positive) && (cc < 0.0)) tot++;
    }
    tot /= (double) number_examples_total;
    return tot;
  }

  public void generate_stratified_sample_with_check(boolean check_labels) {
    // stratifies sample and checks that each training sample has at least one example of each class
    // sign & samples are non trivial
    boolean check_ok = true;
    int i;
    Vector cts;
    double v;

    System.out.print(
        "Generating " + Dataset.NUMBER_STRATIFIED_CV + "-folds stratified sample ... ");

    if (!check_labels) generate_stratified_sample();
    else {
      do {
        if (Debug)
          System.out.print(
              "Checking that each fold has at least one example of each class & is non trivial (no"
                  + " variable with edges of the same sign) ");
        check_ok = true;
        generate_stratified_sample();
        i = 0;
        do {
          cts = (Vector) training_sample.elementAt(i);
          v = getProportionExamplesSign(cts, true);
          if ((v == 0.0) || (v == 1.0)) check_ok = false;

          i++;
          if (Debug) System.out.print(".");
        } while ((i < training_sample.size()) && (check_ok));
        if (Debug && check_ok) System.out.println("ok.");
        if (Debug && !check_ok) System.out.println("\nBad fold# " + (i - 1) + " Retrying");
      } while (!check_ok);
    }
    System.out.print(" ok.\n");
  }

  public void generate_stratified_sample() {
    Vector all = new Vector();
    Vector all2 = new Vector();
    Vector dumv, dumvtr, dumvte, refv;
    Random r = new Random();
    int indexex = 0, indexse = 0;
    stratified_sample = new Vector();
    int i, ir, j, k;

    for (i = 0; i < number_examples_total; i++) all.addElement(new Integer(i));

    do {
      if (all.size() > 1) ir = r.nextInt(all.size());
      else ir = 0;

      all2.addElement((Integer) all.elementAt(ir));
      all.removeElementAt(ir);
    } while (all.size() > 0);

    for (i = 0; i < Dataset.NUMBER_STRATIFIED_CV; i++) stratified_sample.addElement(new Vector());

    do {
      dumv = (Vector) stratified_sample.elementAt(indexse);
      dumv.addElement((Integer) all2.elementAt(indexex));
      indexex++;
      indexse++;
      if (indexse >= Dataset.NUMBER_STRATIFIED_CV) indexse = 0;
    } while (indexex < number_examples_total);

    training_sample = new Vector();
    test_sample = new Vector();

    for (i = 0; i < Dataset.NUMBER_STRATIFIED_CV; i++) {
      dumvtr = new Vector();
      dumvte = new Vector();
      for (j = 0; j < Dataset.NUMBER_STRATIFIED_CV; j++) {
        dumv = (Vector) stratified_sample.elementAt(j);
        if (j == i) refv = dumvte;
        else refv = dumvtr;
        for (k = 0; k < dumv.size(); k++) refv.addElement((Integer) dumv.elementAt(k));
      }
      training_sample.addElement(dumvtr);
      test_sample.addElement(dumvte);
    }
  }

  public int train_size(int fold) {
    return ((Vector) training_sample.elementAt(fold)).size();
  }

  public int test_size(int fold) {
    return ((Vector) test_sample.elementAt(fold)).size();
  }

  public Example train_example(int fold, int nex) {
    return (Example)
        examples.elementAt(
            ((Integer) ((Vector) training_sample.elementAt(fold)).elementAt(nex)).intValue());
  }

  public Example test_example(int fold, int nex) {
    return (Example)
        examples.elementAt(
            ((Integer) ((Vector) test_sample.elementAt(fold)).elementAt(nex)).intValue());
  }

  public void init_weights(String loss_name, int fold, double tt) {
    Example ee;
    int i;
    for (i = 0; i < train_size(fold); i++) {
      ee = train_example(fold, i);
      if (loss_name.equals(Boost.KEY_NAME_TEMPERED_LOSS))
        ee.current_boosting_weight =
            1.0 / Math.pow((double) train_size(fold), 1.0 / (2.0 - tt)); // TEMPERED ADABOOST
      else if (loss_name.equals(Boost.KEY_NAME_LOG_LOSS))
        ee.current_boosting_weight = 0.5; // LOGISTIC / LOG LOSS
    }
  }

  public double averageTrain_size() {
    int i;
    double val = 0.0;
    for (i = 0; i < NUMBER_STRATIFIED_CV; i++) val += (double) train_size(i);
    val /= (double) NUMBER_STRATIFIED_CV;
    return val;
  }
}
