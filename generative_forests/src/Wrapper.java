// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

class Wrapper implements Debuggable {

  public static boolean FAST_SPLITTING = true;

  public static String BOTH_IMPUTE_FILES = "BOTH_IMPUTE_FILES",
      IMPUTE_CSV = "IMPUTE_CSV",
      IMPUTE_TXT = "IMPUTE_TXT";
  public static String DATASET = "--dataset=",
      DATASET_SPEC = "--dataset_spec=",
      DATASET_TEST = "--dataset_test=",
      NUM_SAMPLES = "--num_samples=",
      WORK_DIR = "--work_dir=",
      OUTPUT_SAMPLES = "--output_samples=",
      OUTPUT_STATS = "--output_stats=",
      FLAGS = "--flags",
      HELP = "--help",
      IMPUTE_MISSING = "--impute_missing=",
      DENSITY_ESTIMATION = "--density_estimation=",
      PLOT_LABELS = "--plot_labels=";

  public static String ALGORITHM_CATEGORY = "--algorithm_category=";

  public static String[] ALL_FLAGS = {
    "iterations",
    "unknown_value_coding",
    "force_integer_coding",
    "force_binary_coding",
    "initial_number_of_trees",
    "splitting_method",
    "type_of_generative_model",
    "plot_type",
    "imputation_method"
  };

  public static String GENERATIVE_FOREST = "generative_forest",
      ENSEMBLE_OF_GENERATIVE_TREES = "ensemble_of_generative_trees",
      TRY_THEM_ALL = "try_all";
  public static String PLOT_TYPE_ALL = "all", PLOT_TYPE_DATA = "data";

  public static String BOOSTING = "boosting";

  public static String IMPUTATION_AT_THE_MAX = "at_max_density",
      IMPUTATION_SAMPLING_DENSITY = "sampling_density";

  public static String[] TYPE_OF_GENERATIVE_MODEL = {
    GENERATIVE_FOREST, ENSEMBLE_OF_GENERATIVE_TREES, TRY_THEM_ALL
  };
  public static String[] TYPE_OF_SPLITTING_METHOD = {BOOSTING};
  public static String[] TYPE_OF_IMPUTATION = {
    IMPUTATION_AT_THE_MAX, IMPUTATION_SAMPLING_DENSITY, TRY_THEM_ALL
  };

  public static void CHECK_IMPUTATION_METHOD(String s) {
    int i = 0;
    do {
      if (TYPE_OF_IMPUTATION[i].equals(s)) return;
      else i++;
    } while (i < TYPE_OF_IMPUTATION.length);
    Dataset.perror("Wrapper.class :: no imputation method named " + s);
  }

  public static void CHECK_SPLITTING_METHOD(String s) {
    int i = 0;
    do {
      if (TYPE_OF_SPLITTING_METHOD[i].equals(s)) return;
      else i++;
    } while (i < TYPE_OF_SPLITTING_METHOD.length);
    Dataset.perror("Wrapper.class :: no splitting method named " + s);
  }

  public static void CHECK_GENERATIVE_MODEL(String s) {
    int i = 0;
    do {
      if (TYPE_OF_GENERATIVE_MODEL[i].equals(s)) return;
      else i++;
    } while (i < TYPE_OF_GENERATIVE_MODEL.length);
    Dataset.perror("Wrapper.class :: no generative model named " + s);
  }

  public int INDEX_OF(String s) {
    int i = 0;
    do {
      if (s.equals(TYPE_OF_GENERATIVE_MODEL[i])) return i;
      else i++;
    } while (i < TYPE_OF_GENERATIVE_MODEL.length);
    Dataset.perror("Wrapper.class :: no such use of generative model as " + s);
    return -1;
  }

  // all flag names recognized in command line in --flags = {"name" : value, ...}
  // unknown_value_coding: String = enforces an "unknown value" different from default
  // force_integer_coding: boolean = if true, enforce integer coding of observation variables
  // recognizable as integers ("cleaner" GT)
  // generative_forest = if true, the set of trees is used as generative_forest, otherwise as
  // ensemble_of_generative_trees

  public static int ALL_FLAGS_INDEX_ITERATIONS = 0,
      ALL_FLAGS_INDEX_UNKNOWN_VALUE_CODING = 1,
      ALL_FLAGS_FORCE_INTEGER_CODING = 2,
      ALL_FLAGS_FORCE_BINARY_CODING = 3,
      ALL_FLAGS_INDEX_INITIAL_NUMBER_OF_TREES = 4,
      ALL_FLAGS_SPLITTING_METHOD = 5,
      ALL_FLAGS_TYPE_OF_GENERATIVE_MODEL = 6,
      ALL_FLAGS_PLOT_TYPE = 7,
      ALL_FLAGS_IMPUTATION_METHOD = 8;

  public static String[] DATASET_TOKENS = {
    "\"name\":", "\"path\":", "\"label\":", "\"task\":"
  }; // spec_name, spec_path, spec_label, spec_task

  public static String PREFIX_GENERATOR = "generator_";

  public String path_and_name_of_domain_dataset,
      path_and_name_of_test_dataset = null,
      path_name_test = null,
      path_to_generated_samples,
      working_directory,
      blueprint_save_name,
      spec_name,
      prefix_domain,
      spec_path,
      spec_label,
      spec_task,
      output_stats_file,
      output_stats_directory,
      generator_filename,
      token_save_string,
      type_of_generative_model = Wrapper.GENERATIVE_FOREST,
      splitting_method = Wrapper.BOOSTING,
      plot_type = Wrapper.PLOT_TYPE_DATA,
      imputation_type = Wrapper.IMPUTATION_AT_THE_MAX;
  // spec_name = prefix name

  public String[] flags_values;

  int size_generated, number_iterations; // was nums
  // int index_x_name, index_y_name;

  int[] plot_labels_indexes;
  String[] plot_labels_names;

  String[][] densityplot_filename,
      frontierplot_filename,
      jointdensityplot_filename,
      domaindensityplot_filename,
      generateddatadensityplot_filename,
      imputeddataplot_token;
  int algorithm_category, initial_nb_of_trees;

  Algorithm myAlgos;
  Domain myDomain;
  boolean force_integer_coding = false,
      force_binary_coding = true,
      impute_missing = false,
      density_estimation = false,
      has_missing_values,
      generative_forest = true;
  long loading_time,
      gt_computation_time,
      marginal_computation_time,
      saving_generator_time,
      saving_stats_time,
      generate_observations_time,
      saving_generated_sample_time,
      saving_density_plot_generated_sample_time,
      imputation_time,
      density_estimation_time;

  Wrapper() {
    flags_values = new String[ALL_FLAGS.length];
    size_generated = number_iterations = algorithm_category = -1;
    densityplot_filename =
        frontierplot_filename =
            jointdensityplot_filename =
                domaindensityplot_filename = generateddatadensityplot_filename = null;
    token_save_string = null;
    plot_labels_names = null;
    plot_labels_indexes = null;
    path_and_name_of_domain_dataset = spec_path = null;

    loading_time =
        gt_computation_time =
            marginal_computation_time =
                saving_generator_time =
                    saving_stats_time =
                        generate_observations_time =
                            saving_generated_sample_time =
                                saving_density_plot_generated_sample_time = 0;
    has_missing_values = false;
  }

  public static String help() {
    String ret = "";

    ret += "Example run\n";
    ret += "Java Wrapper --dataset=${ANYDIR}/Datasets/iris/iris.csv\n";
    ret +=
        "             '--dataset_spec={\"name\": \"iris\", \"path\":"
            + " \"${ANYDIR}/Datasets/iris/iris.csv\", \"label\": \"class\", \"task\":"
            + " \"BINARY_CLASSIFICATION\"}'\n";
    ret += "             --num_samples=1000 \n";
    ret += "             --work_dir=${ANYDIR}/Datasets/iris/working_dir \n";
    ret +=
        "            "
            + " --output_samples=${ANYDIR}/Datasets/iris/output_samples/iris_geot_generated.csv\n";
    ret +=
        "             --output_stats=${ANYDIR}/Datasets/iris/results/generated_observations.stats"
            + " \n";
    ret += "             '--plot_labels={\"Sepal.Length\",\"Sepal.Width\"}' \n";
    ret +=
        "             '--flags={\"iterations\" : \"10\", \"force_integer_coding\" : \"true\","
            + " \"force_binary_coding\" : \"true\", \"unknown_value_coding\" : \"NA\","
            + " \"initial_number_of_trees\" : \"2\", \"splitting_method\" : \"boosting\","
            + " \"type_of_generative_model\" : \"try_all\"}'\n";
    ret += "             --impute_missing=true\n\n";

    ret += " --dataset: path to access the.csv data file containing variable names in first line\n";
    ret +=
        " --dataset_test: path to access the.csv data file containing the test dataset (for density"
            + " estimation)\n";
    ret += " --dataset_spec: self explanatory\n";
    ret += " --num_samples: number of generated observations\n";
    ret += " --work_dir: directory where the generative models and density plots are saved\n";
    ret += " --output_samples: generated samples filename\n";
    ret +=
        " --output_stats: file to store all data related to run (execution times, node stats in"
            + " tree(s), etc)\n";
    ret +=
        " --plot_labels: (optional) variables used to save 2D plots of (i) frontiers, (ii) density"
            + " learned, (iii) domain density\n";
    ret +=
        "                use >= 2 features, CONTINUOUS only & small dims / domain size for"
            + " efficient computation (otherwise, can be very long)\n";
    ret += " --flags: flags...\n";
    ret += "          iterations (mandatory): integer; number of *global* boosting iterations\n";
    ret +=
        "          force_integer_coding (optional): boolean; if true, recognizes integer variables"
            + " and codes them as such (otherwise, codes them as doubles) -- default: false\n";
    ret +=
        "          force_binary_coding (optional): boolean; if true, recognizes 0/1/unknown"
            + " variables and codes them as nominal, otherwise treat them as integers or doubles --"
            + " default: true\n";
    ret +=
        "          unknown_value_coding (optional): String; representation of 'unknown' value in"
            + " dataset -- default: \"-1\"\n";
    ret += "          initial_number_of_trees (mandatory): integer; number of trees in the model\n";
    ret += "          splitting_method (mandatory): String in {\"boosting\", \"random\"}\n";
    ret +=
        "                                        method used to split nodes among two possible:\n";
    ret +=
        "                                              boosting : use boost using methods described"
            + " in the draft\n";
    ret += "                                              random : random splits\n";
    ret +=
        "          type_of_generative_model (mandatory): String in {\"generative_forest\","
            + " \"ensemble_of_generative_trees\", \"try_all\"}\n";
    ret +=
        "                                                what kind of generative model learned /"
            + " used among three possible:\n";
    ret +=
        "                                                   generative_forest : generative ensemble"
            + " of trees (uses training sample to compute probabilities)\n";
    ret +=
        "                                                   ensemble_of_generative_trees : ensemble"
            + " of generative trees (uses only generative trees w/ probabilities at the arcs)\n";
    ret += "                                                   try_all : loop over both types\n";
    ret += "          plot_type (optional): String in {\"all\", \"data\"}\n";
    ret +=
        "                                plotting types (if variables specified with"
            + " plot_labels):\n";
    ret +=
        "                                        all : maximum # of plots (domain, generated,"
            + " frontiers of density learned, density learned) -- can be time consuming\n";
    ret +=
        "                                        data : only domain and generated data plots (if"
            + " available)\n";
    ret +=
        "          imputation_method (optional): String in {\"at_max_density\","
            + " \"using_distribution\"}\n";
    ret +=
        "                                        method used to impute data based on generative"
            + " model\n";
    ret +=
        "                                               at_max_density : imputes at peak density"
            + " conditioned to missing features\n";
    ret +=
        "                                               sampling_density : imputes by sampling"
            + " density conditioned to missing features\n";
    ret += "                                               try_all : loop over both types\n";
    ret +=
        " --impute_missing: if true, uses the generated tree to impute the missing values in the"
            + " training data\n\n";
    ret +=
        " --density_estimation: if true, performs density estimation at each step of training,"
            + " using the dataset dataset_test.csv \n\n";
    ret +=
        " * warning * : parameters may be included in file names to facilitate identification of"
            + " training parameters, on the form of a token SW_IX_TY_GEOTZ\n";
    ret +=
        "               W in {0,1} = 1 iff split type is boosting, X = # iterations, Y = # trees, Z"
            + " in {0,1} = 1 iff generative ensemble of trees learned / used\n";

    return ret;
  }

  public static String DUMS =
      "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////";

  public static String FILL(String s) {
    String basics = "// Training GF / EOGT () //";
    int i;
    String ret = "";
    for (i = 0; i < s.length() - basics.length() - History.CURRENT_HISTORY().length() - 1; i++)
      ret += " ";
    return ret;
  }

  public static void main(String[] arg) {
    int i, j;
    Wrapper w = new Wrapper();

    System.out.println("");
    System.out.println(Wrapper.DUMS);
    System.out.println(
        "// Training GF / EOGT (" + History.CURRENT_HISTORY() + ") " + FILL(Wrapper.DUMS) + " //");
    System.out.println(
        "//                                                                                        "
            + "                                //");

    if (arg.length == 0) {
      System.out.println("// *No parameters*. Run 'java Wrapper --help' for more");
      System.exit(0);
    }

    System.out.println(
        "// Help & observation run: 'java Wrapper --help'                                          "
            + "                                //");
    System.out.println(
        "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////");

    for (i = 0; i < arg.length; i++) {
      if (arg[i].equals(HELP)) {
        System.out.println(help());
        System.exit(0);
      }
      w.fit_vars(arg[i]);
    }

    w.summary();
    w.simple_go();
  }

  public void simple_go() {
    long b, e;
    int i;
    Vector<Observation> v_gen = null;
    GenerativeModelBasedOnEnsembleOfTrees geot;
    boolean try_model;

    String which_model = "dummy", prefix_output_stats_file = output_stats_file;

    System.out.print("Loading stuff start... ");
    b = System.currentTimeMillis();
    myDomain = new Domain(this);
    e = System.currentTimeMillis();
    loading_time = e - b;
    System.out.println("Loading stuff ok (time elapsed: " + loading_time + " ms).\n");

    if ((density_estimation)
        && ((path_and_name_of_test_dataset == null) || (path_and_name_of_test_dataset.equals(""))))
      Dataset.perror("Wrapper.class :: density estimation but no test dataset");

    for (i = 0; i <= 1; i++) {
      try_model = false;
      if ((i == 0)
          && ((type_of_generative_model.equals(Wrapper.GENERATIVE_FOREST))
              || (type_of_generative_model.equals(Wrapper.TRY_THEM_ALL)))) {
        myAlgos =
            new Algorithm(
                myDomain.myDS, 0, number_iterations, initial_nb_of_trees, splitting_method, true);
        which_model = "GF";
        try_model = true;
      } else if ((i == 1)
          && ((type_of_generative_model.equals(Wrapper.ENSEMBLE_OF_GENERATIVE_TREES))
              || (type_of_generative_model.equals(Wrapper.TRY_THEM_ALL)))) {
        myAlgos =
            new Algorithm(
                myDomain.myDS, 0, number_iterations, initial_nb_of_trees, splitting_method, false);
        which_model = "EOGT";
        try_model = true;
      }

      if (try_model) {
        System.out.print("Learning the generator... ");
        b = System.currentTimeMillis();
        geot = myAlgos.learn_geot();
        e = System.currentTimeMillis();
        gt_computation_time = e - b;
        System.out.println("ok (time elapsed: " + gt_computation_time + " ms).\n\n");

        System.out.println(which_model + " used:\n " + geot);

        token_save_string =
            "_S"
                + ((splitting_method.equals(Wrapper.BOOSTING)) ? 1 : 0)
                + "_I"
                + number_iterations
                + "_T"
                + initial_nb_of_trees
                + "_GEOT"
                + ((geot.generative_forest) ? 1 : 0)
                + "_";
        output_stats_file = prefix_output_stats_file + token_save_string;

        compute_filenames();

        save_model(geot);

        if (size_generated > 0) {
          System.out.print(
              "Generating " + size_generated + " observations using the " + which_model + "... ");
          b = System.currentTimeMillis();
          v_gen = geot.generate_sample_with_density(size_generated);
          e = System.currentTimeMillis();
          generate_observations_time = e - b;
          System.out.println("ok (time elapsed: " + generate_observations_time + " ms).");
        }

        if (plot_labels_names != null) {
          System.out.print("Saving projected frontiers & density plots... ");
          b = System.currentTimeMillis();
          save_frontier_and_density_plot(geot, v_gen);
          e = System.currentTimeMillis();
          System.out.println("ok (time elapsed: " + (e - b) + " ms).");
        }

        if ((has_missing_values) && (impute_missing)) {
          System.out.print("Imputing observations using the " + which_model + "... ");
          b = System.currentTimeMillis();
          // impute_and_save(geot, true);
          impute_and_save(geot, Wrapper.BOTH_IMPUTE_FILES);
          e = System.currentTimeMillis();
          imputation_time = e - b;
          System.out.println("ok (time elapsed: " + imputation_time + " ms).");
        }

        if (density_estimation) {
          System.out.print("Density estimation using the " + which_model + "... ");
          density_estimation_save(myAlgos);
          System.out.println("ok.");
        }

        System.out.print("Saving generated sample... ");
        b = System.currentTimeMillis();
        save_sample(v_gen);
        e = System.currentTimeMillis();
        saving_generated_sample_time = e - b;
        System.out.println("ok (time elapsed: " + saving_generated_sample_time + " ms).");

        System.out.print("Saving stats file... ");
        b = System.currentTimeMillis();
        save_stats(geot);
        e = System.currentTimeMillis();
        saving_stats_time = e - b;
        System.out.println("ok (time elapsed: " + saving_stats_time + " ms).");
      }
    }

    System.out.println("All finished. Stopping...");
    myDomain.myMemoryMonitor.stop();
  }

  public void save_sample(Vector<Observation> v_gen) {
    FileWriter f;
    int i;

    String save_name;
    if (blueprint_save_name.contains(".csv"))
      save_name =
          blueprint_save_name.substring(0, blueprint_save_name.lastIndexOf(".csv"))
              + token_save_string
              + ".csv";
    else save_name = blueprint_save_name + token_save_string + ".csv";

    String nameFile = path_to_generated_samples + "/" + save_name;

    try {
      f = new FileWriter(nameFile);

      for (i = 0; i < myDomain.myDS.features_names_from_file.length; i++) {
        f.write(myDomain.myDS.features_names_from_file[i]);
        if (i < myDomain.myDS.features_names_from_file.length - 1)
          f.write(Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
      }
      f.write("\n");

      for (i = 0; i < v_gen.size(); i++)
        f.write(((Observation) v_gen.elementAt(i)).toStringSave(true) + "\n");

      f.close();
    } catch (IOException e) {
      Dataset.perror("Experiments.class :: Saving results error in file " + nameFile);
    }
  }

  public void save_density_plot_sample(Vector<Observation> v_gen, boolean in_csv) {
    FileWriter f;
    int i, j, k;
    String nameFile;

    for (i = 0; i < plot_labels_names.length - 1; i++) {
      for (j = i + 1; j < plot_labels_names.length; j++) {
        nameFile = working_directory + "/" + densityplot_filename[i][j];
        if (!in_csv) nameFile += ".txt";

        try {
          f = new FileWriter(nameFile);

          if (in_csv)
            f.write("//" + plot_labels_names[i] + "," + plot_labels_names[j] + ",density_value\n");

          for (k = 0; k < v_gen.size(); k++)
            f.write(
                ((Observation) v_gen.elementAt(k))
                        .toStringSaveDensity(plot_labels_indexes[i], plot_labels_indexes[j], in_csv)
                    + "\n");

          f.close();
        } catch (IOException e) {
          Dataset.perror("Experiments.class :: Saving results error in file " + nameFile);
        }
      }
    }
  }

  public void save_model(GenerativeModelBasedOnEnsembleOfTrees geot) {
    FileWriter f;
    String nameFile;
    nameFile = working_directory + "/" + spec_name + token_save_string + "_generative_model.txt";
    try {
      f = new FileWriter(nameFile);

      f.write(geot.toString());

      f.close();
    } catch (IOException e) {
      Dataset.perror("Experiments.class :: Saving results error in file " + nameFile);
    }
  }

  public void impute_and_save(GenerativeModelBasedOnEnsembleOfTrees geot, String which_file) {
    FileWriter f;
    int i, j, k;

    Vector<Observation> imputed_observations_at_the_max = null;
    Vector<Observation> imputed_observations_sampling_density = null;
    Vector<Observation> imputed_observations_target;

    if ((imputation_type.equals(Wrapper.TRY_THEM_ALL))
        || (imputation_type.equals(Wrapper.IMPUTATION_AT_THE_MAX)))
      imputed_observations_at_the_max = new Vector<>();

    if ((imputation_type.equals(Wrapper.TRY_THEM_ALL))
        || (imputation_type.equals(Wrapper.IMPUTATION_SAMPLING_DENSITY)))
      imputed_observations_sampling_density = new Vector<>();

    String nameFile, token_imputation_type;

    Observation ee, ee_cop_at_the_max, ee_cop_sampling_density;

    for (i = 0; i < myDomain.myDS.observations_from_file.size(); i++) {

      ee_cop_at_the_max = ee_cop_sampling_density = null;

      if ((i % (myDomain.myDS.observations_from_file.size() / 100) == 0)
          && ((i / (myDomain.myDS.observations_from_file.size() / 100)) <= 100))
        System.out.print((i / (myDomain.myDS.observations_from_file.size() / 100)) + "% ");
      ee = (Observation) myDomain.myDS.observations_from_file.elementAt(i);

      if ((imputation_type.equals(Wrapper.TRY_THEM_ALL))
          || (imputation_type.equals(Wrapper.IMPUTATION_AT_THE_MAX)))
        ee_cop_at_the_max = Observation.copyOf(ee);

      if ((imputation_type.equals(Wrapper.TRY_THEM_ALL))
          || (imputation_type.equals(Wrapper.IMPUTATION_SAMPLING_DENSITY)))
        ee_cop_sampling_density = Observation.copyOf(ee);

      if (ee.contains_unknown_values()) {
        geot.impute(ee_cop_at_the_max, ee_cop_sampling_density, i);
      }

      if ((imputation_type.equals(Wrapper.TRY_THEM_ALL))
          || (imputation_type.equals(Wrapper.IMPUTATION_AT_THE_MAX)))
        imputed_observations_at_the_max.addElement(ee_cop_at_the_max);

      if ((imputation_type.equals(Wrapper.TRY_THEM_ALL))
          || (imputation_type.equals(Wrapper.IMPUTATION_SAMPLING_DENSITY)))
        imputed_observations_sampling_density.addElement(ee_cop_sampling_density);
    }

    if (plot_labels_names != null) {
      Plotting my_plot;

      System.out.print(" (Imputed data plots");
      for (k = 0; k < 2; k++) {
        imputed_observations_target = null;
        token_imputation_type = null;
        if ((k == 0)
            && ((imputation_type.equals(Wrapper.IMPUTATION_AT_THE_MAX))
                || (imputation_type.equals(Wrapper.TRY_THEM_ALL)))) {
          imputed_observations_target = imputed_observations_at_the_max;
          token_imputation_type = Wrapper.IMPUTATION_AT_THE_MAX;
        } else if ((k == 1)
            && ((imputation_type.equals(Wrapper.IMPUTATION_SAMPLING_DENSITY))
                || (imputation_type.equals(Wrapper.TRY_THEM_ALL)))) {
          imputed_observations_target = imputed_observations_sampling_density;
          token_imputation_type = Wrapper.IMPUTATION_SAMPLING_DENSITY;
        }
        if (imputed_observations_target != null) {
          for (i = 0; i < plot_labels_names.length; i++) {
            for (j = 0; j < plot_labels_names.length; j++) {
              if (j != i) {
                System.out.print(" [" + plot_labels_names[i] + ":" + plot_labels_names[j] + "]");
                nameFile =
                    working_directory
                        + "/"
                        + imputeddataplot_token[i][j]
                        + "_"
                        + token_imputation_type
                        + ".png";

                my_plot = new Plotting();
                my_plot.init(geot, plot_labels_indexes[i], plot_labels_indexes[j]);
                my_plot.compute_and_store_x_y_densities_dataset(
                    geot, imputed_observations_target, nameFile);
              }
            }
          }
        }
      }
      System.out.print(") ");
    }

    if ((which_file.equals(Wrapper.BOTH_IMPUTE_FILES)) || (which_file.equals(Wrapper.IMPUTE_CSV))) {
      for (k = 0; k < 2; k++) {
        imputed_observations_target = null;
        token_imputation_type = null;
        if ((k == 0)
            && ((imputation_type.equals(Wrapper.IMPUTATION_AT_THE_MAX))
                || (imputation_type.equals(Wrapper.TRY_THEM_ALL)))) {
          imputed_observations_target = imputed_observations_at_the_max;
          token_imputation_type = Wrapper.IMPUTATION_AT_THE_MAX;
        } else if ((k == 1)
            && ((imputation_type.equals(Wrapper.IMPUTATION_SAMPLING_DENSITY))
                || (imputation_type.equals(Wrapper.TRY_THEM_ALL)))) {
          imputed_observations_target = imputed_observations_sampling_density;
          token_imputation_type = Wrapper.IMPUTATION_SAMPLING_DENSITY;
        }
        if (imputed_observations_target != null) {
          nameFile =
              working_directory
                  + "/"
                  + spec_name
                  + token_save_string
                  + "_"
                  + token_imputation_type
                  + "_imputed.csv";
          try {
            f = new FileWriter(nameFile);

            for (i = 0; i < myDomain.myDS.features_names_from_file.length; i++) {
              f.write(myDomain.myDS.features_names_from_file[i]);
              if (i < myDomain.myDS.features_names_from_file.length - 1)
                f.write(Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
            }
            f.write("\n");

            for (i = 0; i < imputed_observations_target.size(); i++)
              f.write(imputed_observations_target.elementAt(i).toStringSave(true) + "\n");

            f.close();
          } catch (IOException e) {
            Dataset.perror("Experiments.class :: Saving results error in file " + nameFile);
          }
        }
      }
    }

    if ((which_file.equals(Wrapper.BOTH_IMPUTE_FILES)) || (which_file.equals(Wrapper.IMPUTE_TXT))) {
      for (k = 0; k < 2; k++) {
        imputed_observations_target = null;
        token_imputation_type = null;
        if ((k == 0)
            && ((imputation_type.equals(Wrapper.IMPUTATION_AT_THE_MAX))
                || (imputation_type.equals(Wrapper.TRY_THEM_ALL)))) {
          imputed_observations_target = imputed_observations_at_the_max;
          token_imputation_type = Wrapper.IMPUTATION_AT_THE_MAX;
        } else if ((k == 1)
            && ((imputation_type.equals(Wrapper.IMPUTATION_SAMPLING_DENSITY))
                || (imputation_type.equals(Wrapper.TRY_THEM_ALL)))) {
          imputed_observations_target = imputed_observations_sampling_density;
          token_imputation_type = Wrapper.IMPUTATION_SAMPLING_DENSITY;
        }
        if (imputed_observations_target != null) {
          nameFile =
              working_directory
                  + "/"
                  + spec_name
                  + token_save_string
                  + "_"
                  + token_imputation_type
                  + "_imputed.csv.txt";
          try {
            f = new FileWriter(nameFile);

            f.write("#");
            for (i = 0; i < myDomain.myDS.features_names_from_file.length; i++) {
              f.write(myDomain.myDS.features_names_from_file[i]);
              if (i < myDomain.myDS.features_names_from_file.length - 1)
                f.write(Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
            }
            f.write("\n");

            for (i = 0; i < imputed_observations_target.size(); i++)
              f.write(imputed_observations_target.elementAt(i).toStringSave(false) + "\n");

            f.close();
          } catch (IOException e) {
            Dataset.perror("Experiments.class :: Saving results error in file " + nameFile);
          }
        }
      }
    }
  }

  public void save_frontier_and_density_plot(
      GenerativeModelBasedOnEnsembleOfTrees geot, Vector<Observation> v_gen) {

    String fpname;
    String dpname;
    String domain_dpname;
    String generated_dpname;
    int i, j;
    Plotting my_plot;

    if (plot_labels_names != null) {
      for (i = 0; i < plot_labels_names.length; i++) {
        for (j = 0; j < plot_labels_names.length; j++) {
          if (j != i) {
            System.out.print("[" + plot_labels_names[i] + ":" + plot_labels_names[j] + "]");

            fpname = working_directory + "/" + frontierplot_filename[i][j];
            dpname = working_directory + "/" + jointdensityplot_filename[i][j];
            domain_dpname = working_directory + "/" + domaindensityplot_filename[i][j];

            if (plot_type.equals(Wrapper.PLOT_TYPE_ALL)) {
              System.out.print("(frontiers)");
              my_plot = new Plotting();
              my_plot.init(geot, plot_labels_indexes[i], plot_labels_indexes[j]);
              my_plot.compute_and_store_x_y_frontiers(geot, fpname);

              if (geot.generative_forest) {
                System.out.print("(densities)");
                my_plot = new Plotting();
                my_plot.init(geot, plot_labels_indexes[i], plot_labels_indexes[j]);
                my_plot.compute_and_store_x_y_densities(geot, dpname);
              }
            }

            if ((plot_type.equals(Wrapper.PLOT_TYPE_ALL))
                || (plot_type.equals(Wrapper.PLOT_TYPE_DATA))) {
              System.out.print("(domain)");
              my_plot = new Plotting();
              my_plot.init(geot, plot_labels_indexes[i], plot_labels_indexes[j]);
              my_plot.compute_and_store_x_y_densities_dataset(
                  geot, geot.myDS.observations_from_file, domain_dpname);

              if (v_gen != null) {
                generated_dpname =
                    working_directory
                        + "/"
                        + generateddatadensityplot_filename[i][j]
                        + "_"
                        + v_gen.size()
                        + ".png";
                System.out.print("(generated)");
                my_plot.init(geot, plot_labels_indexes[i], plot_labels_indexes[j]);
                my_plot.compute_and_store_x_y_densities_dataset(geot, v_gen, generated_dpname);
              }
            }
          }
        }
      }
    }
  }

  public void save_stats(GenerativeModelBasedOnEnsembleOfTrees geot) {
    int i, j;
    FileWriter f = null;
    int[] node_counts = new int[myDomain.myDS.features_names_from_file.length];
    String output_stats_additional = output_stats_file + "_more.txt";

    long time_gt_plus_generation = gt_computation_time + generate_observations_time;
    long time_gt_plus_imputation = -1;
    if ((has_missing_values) && (impute_missing))
      time_gt_plus_imputation = gt_computation_time + imputation_time;

    String all_depths = "{";
    String all_number_nodes = "{";

    for (i = 0; i < geot.trees.size(); i++) {
      all_depths += geot.trees.elementAt(i).depth + "";
      all_number_nodes += geot.trees.elementAt(i).number_nodes + "";
      if (i < geot.trees.size() - 1) {
        all_depths += ",";
        all_number_nodes += ",";
      }
      geot.trees.elementAt(i).root.recursive_fill_node_counts(node_counts);
    }

    all_depths += "}";
    all_number_nodes += "}";

    try {
      f = new FileWriter(output_stats_file + ".txt");

      f.write("{\n");

      f.write(
          " \"running_time_seconds\": "
              + DF8.format(((double) gt_computation_time + generate_observations_time) / 1000)
              + ",\n");
      f.write(" \"geot_eogt_number_trees\": " + geot.trees.size() + ",\n");
      f.write(" \"geot_eogt_nodes_per_tree\": " + all_number_nodes + ",\n");
      f.write(" \"geot_eogt_depth_per_tree\": " + all_depths + ",\n");
      f.write(
          " \"running_time_training_plus_exemple_generation\": "
              + DF8.format(((double) time_gt_plus_generation / 1000))
              + ",\n");
      if ((has_missing_values) && (impute_missing))
        f.write(
            " \"running_time_training_plus_imputation\": "
                + DF8.format(((double) time_gt_plus_imputation / 1000))
                + "\n");

      f.write("}\n");

      f.close();
    } catch (IOException e) {
      Dataset.perror("Wrapper.class :: Saving results error in file " + output_stats_file);
    }

    try {
      f = new FileWriter(output_stats_additional);

      f.write("// flag values used: ");
      for (i = 0; i < flags_values.length; i++)
        f.write(" [" + ALL_FLAGS[i] + ":" + flags_values[i] + "] ");
      f.write("\n\n");

      f.write("// Time to learn the generator: " + gt_computation_time + " ms.\n");
      f.write("// Time to generate sample: " + generate_observations_time + " ms.\n");

      f.write("\n// Generator: overall node counts per feature name (w/ interpreted type)\n");
      for (i = 0; i < myDomain.myDS.features_names_from_file.length; i++)
        f.write(
            "// "
                + myDomain.myDS.features.elementAt(i).name
                + " ("
                + myDomain.myDS.features.elementAt(i).type
                + ") : "
                + node_counts[i]
                + "\n");

      f.write("\n// Generator: per-tree node counts per feature name\n");
      for (i = 0; i < geot.trees.size(); i++) {
        f.write("// Tree# " + i + " :\t");
        for (j = 0;
            j < geot.trees.elementAt(i).statistics_number_of_nodes_for_each_feature.length;
            j++)
          f.write(geot.trees.elementAt(i).statistics_number_of_nodes_for_each_feature[j] + "\t");
        f.write("\n");
      }

      f.close();
    } catch (IOException e) {
      Dataset.perror("Wrapper.class :: Saving results error in file " + output_stats_additional);
    }
  }

  public void compute_filenames() {
    int i, j;

    if (plot_labels_names != null) {
      densityplot_filename = new String[plot_labels_names.length][];
      frontierplot_filename = new String[plot_labels_names.length][];
      jointdensityplot_filename = new String[plot_labels_names.length][];
      domaindensityplot_filename = new String[plot_labels_names.length][];
      generateddatadensityplot_filename = new String[plot_labels_names.length][];
      imputeddataplot_token = new String[plot_labels_names.length][];

      for (i = 0; i < plot_labels_names.length; i++) {
        densityplot_filename[i] = new String[plot_labels_names.length];
        frontierplot_filename[i] = new String[plot_labels_names.length];
        jointdensityplot_filename[i] = new String[plot_labels_names.length];
        domaindensityplot_filename[i] = new String[plot_labels_names.length];
        generateddatadensityplot_filename[i] = new String[plot_labels_names.length];
        imputeddataplot_token[i] = new String[plot_labels_names.length];

        for (j = 0; j < plot_labels_names.length; j++) {
          if (j != i) {
            densityplot_filename[i][j] =
                blueprint_save_name.substring(0, blueprint_save_name.lastIndexOf('.'))
                    + token_save_string
                    + "_X_"
                    + plot_labels_names[i]
                    + "_Y_"
                    + plot_labels_names[j]
                    + blueprint_save_name.substring(
                        blueprint_save_name.lastIndexOf('.'), blueprint_save_name.length())
                    + "_2DDensity_plot";
            frontierplot_filename[i][j] =
                blueprint_save_name.substring(0, blueprint_save_name.lastIndexOf('.'))
                    + token_save_string
                    + "_X_"
                    + plot_labels_names[i]
                    + "_Y_"
                    + plot_labels_names[j]
                    + "_projectedfrontiers_plot.png";
            jointdensityplot_filename[i][j] =
                blueprint_save_name.substring(0, blueprint_save_name.lastIndexOf('.'))
                    + token_save_string
                    + "_X_"
                    + plot_labels_names[i]
                    + "_Y_"
                    + plot_labels_names[j]
                    + "_jointdensity_plot.png";
            domaindensityplot_filename[i][j] =
                blueprint_save_name.substring(0, blueprint_save_name.lastIndexOf('.'))
                    + token_save_string
                    + "_X_"
                    + plot_labels_names[i]
                    + "_Y_"
                    + plot_labels_names[j]
                    + "_jointdensity_plot_domaindensity.png";
            generateddatadensityplot_filename[i][j] =
                blueprint_save_name.substring(0, blueprint_save_name.lastIndexOf('.'))
                    + token_save_string
                    + "_X_"
                    + plot_labels_names[i]
                    + "_Y_"
                    + plot_labels_names[j]
                    + "_jointdensity_plot_generated";
            imputeddataplot_token[i][j] =
                blueprint_save_name.substring(0, blueprint_save_name.lastIndexOf('.'))
                    + token_save_string
                    + "_X_"
                    + plot_labels_names[i]
                    + "_Y_"
                    + plot_labels_names[j]
                    + "_jointdensity_plot_imputed";
          }
        }
      }
    }
  }

  public void summary() {
    int i, j;

    System.out.println("\nRunning copycat generator training with the following inputs:");
    System.out.println(" * dataset path to train generator:" + path_and_name_of_domain_dataset);
    System.out.println(" * working directory:" + working_directory);
    System.out.println(
        " * generated samples ("
            + size_generated
            + " observations) stored in directory "
            + path_to_generated_samples
            + " with filename "
            + blueprint_save_name);
    System.out.println(
        " * generator (gt) stored in working directory with filename " + generator_filename);
    System.out.println(" * stats file at " + output_stats_file);

    if (spec_path == null) Dataset.warning(" No path information in --dataset_spec");
    else if (!path_and_name_of_domain_dataset.equals(spec_path))
      Dataset.warning(
          " Non identical information in --dataset_spec path vs --dataset"
              + " (path_and_name_of_domain_dataset = "
              + path_and_name_of_domain_dataset
              + ", spec_path = "
              + spec_path
              + ")\n");

    if (impute_missing)
      System.out.println(
          " * imputed sample saved at filename "
              + working_directory
              + "/"
              + spec_name
              + "_imputed.csv");

    System.out.print(" * flags (non null): ");
    for (i = 0; i < flags_values.length; i++)
      if (flags_values[i] != null)
        System.out.print("[" + ALL_FLAGS[i] + ":" + flags_values[i] + "] ");
    System.out.println("");

    if (flags_values[ALL_FLAGS_INDEX_UNKNOWN_VALUE_CODING] != null)
      FeatureValue.S_UNKNOWN = flags_values[ALL_FLAGS_INDEX_UNKNOWN_VALUE_CODING];

    if (flags_values[ALL_FLAGS_FORCE_INTEGER_CODING] != null)
      force_integer_coding = Boolean.parseBoolean(flags_values[ALL_FLAGS_FORCE_INTEGER_CODING]);

    if (flags_values[ALL_FLAGS_FORCE_BINARY_CODING] != null)
      force_binary_coding = Boolean.parseBoolean(flags_values[ALL_FLAGS_FORCE_BINARY_CODING]);

    if (flags_values[ALL_FLAGS_INDEX_ITERATIONS] != null)
      number_iterations = Integer.parseInt(flags_values[ALL_FLAGS_INDEX_ITERATIONS]);

    if (flags_values[ALL_FLAGS_INDEX_INITIAL_NUMBER_OF_TREES] != null)
      initial_nb_of_trees = Integer.parseInt(flags_values[ALL_FLAGS_INDEX_INITIAL_NUMBER_OF_TREES]);

    if (flags_values[ALL_FLAGS_SPLITTING_METHOD] != null)
      splitting_method = new String(flags_values[ALL_FLAGS_SPLITTING_METHOD]);

    Wrapper.CHECK_SPLITTING_METHOD(splitting_method);

    if (flags_values[ALL_FLAGS_TYPE_OF_GENERATIVE_MODEL] != null)
      type_of_generative_model = new String(flags_values[ALL_FLAGS_TYPE_OF_GENERATIVE_MODEL]);

    Wrapper.CHECK_GENERATIVE_MODEL(type_of_generative_model);

    if (flags_values[ALL_FLAGS_PLOT_TYPE] != null)
      plot_type = new String(flags_values[ALL_FLAGS_PLOT_TYPE]);

    if (flags_values[ALL_FLAGS_IMPUTATION_METHOD] != null)
      imputation_type = new String(flags_values[ALL_FLAGS_IMPUTATION_METHOD]);

    // token_save_string = "_A" + algorithm_category + "_I" + number_iterations + "_T" +
    // initial_nb_of_trees + "_GEOT" + ((generative_forest) ? 1 : 0) + "_";
    // output_stats_file += token_save_string;

    if (plot_labels_names != null) {
      System.out.print(
          " * 2D density plots, frontier plots, joint density plots and / or domain density plots"
              + " to be recorded on couples of features in (");
      for (j = 0; j < plot_labels_names.length; j++) {
        System.out.print(plot_labels_names[j]);
        if (j < plot_labels_names.length - 1) System.out.print(",");
      }
      System.out.println(")");
    }

    System.out.println("");
  }

  public void fit_vars(String s) {
    String dummys;

    if (s.contains(DATASET)) {
      path_and_name_of_domain_dataset = s.substring(DATASET.length(), s.length());

      spec_name =
          path_and_name_of_domain_dataset.substring(
              path_and_name_of_domain_dataset.lastIndexOf('/') + 1,
              path_and_name_of_domain_dataset.lastIndexOf('.'));

    } else if (s.contains(DATASET_TEST)) {
      path_and_name_of_test_dataset = s.substring(DATASET_TEST.length(), s.length());
      path_name_test =
          path_and_name_of_test_dataset.substring(
              0, path_and_name_of_test_dataset.lastIndexOf('/') + 1);
    } else if (s.contains(DATASET_SPEC)) {
      spec_path = spec_label = spec_task = null;
      int i, begin_ind, end_ind;

      String[] values = new String[4];
      int[] index_tokens = {0, 0, 0, 0};
      for (i = 0; i < DATASET_TOKENS.length; i++) {
        if (s.indexOf(DATASET_TOKENS[i]) != s.lastIndexOf(DATASET_TOKENS[i]))
          Dataset.perror(
              "Wrapper.class :: more than one occurrence of "
                  + DATASET_TOKENS[i]
                  + " in string"
                  + s);
        if (s.indexOf(DATASET_TOKENS[i]) == -1)
          Dataset.perror(
              "Wrapper.class :: zero occurrence of " + DATASET_TOKENS[i] + " in string" + s);
        else index_tokens[i] = s.indexOf(DATASET_TOKENS[i]);
      }
      for (i = 0; i < DATASET_TOKENS.length - 1; i++)
        if (index_tokens[i] > index_tokens[i + 1])
          Dataset.perror(
              "Wrapper.class :: token "
                  + DATASET_TOKENS[i]
                  + " should appear before token "
                  + DATASET_TOKENS[i + 1]
                  + " in string"
                  + s);
      for (i = 0; i < DATASET_TOKENS.length; i++) {
        begin_ind = index_tokens[i] + DATASET_TOKENS[i].length();
        if (i == DATASET_TOKENS.length - 1) end_ind = s.length();
        else end_ind = index_tokens[i + 1] - 1;

        dummys = s.substring(begin_ind, end_ind);
        values[i] = dummys.substring(dummys.indexOf('\"') + 1, dummys.lastIndexOf('\"'));
      }
      prefix_domain = spec_name;

      spec_path = values[1];
      spec_label = values[2];
      spec_task = values[3];
    } else if (s.contains(NUM_SAMPLES)) {
      size_generated = Integer.parseInt(s.substring(NUM_SAMPLES.length(), s.length()));
    } else if (s.contains(ALGORITHM_CATEGORY)) {
      algorithm_category = Integer.parseInt(s.substring(ALGORITHM_CATEGORY.length(), s.length()));
    } else if (s.contains(WORK_DIR)) {
      working_directory = s.substring(WORK_DIR.length(), s.length());
    } else if (s.contains(OUTPUT_SAMPLES)) {
      dummys = s.substring(OUTPUT_SAMPLES.length(), s.length());
      path_to_generated_samples = dummys.substring(0, dummys.lastIndexOf('/'));
      blueprint_save_name = dummys.substring(dummys.lastIndexOf('/') + 1, dummys.length());
      generator_filename = PREFIX_GENERATOR + blueprint_save_name;
    } else if (s.contains(OUTPUT_STATS)) {
      output_stats_file = s.substring(OUTPUT_STATS.length(), s.length());
      output_stats_directory = s.substring(OUTPUT_STATS.length(), s.lastIndexOf('/'));
    } else if (s.contains(PLOT_LABELS)) {
      String all_labels = s.substring(PLOT_LABELS.length(), s.length()), dums;
      Vector<String> all_strings = new Vector<>();
      boolean in_s = false, word_ok = false;
      int i = 0, i_b = -1, i_e = -1, j;
      do {
        if (all_labels.charAt(i) == '"') {
          if (in_s) {
            i_e = i;
            word_ok = true;
            in_s = false;
          } else {
            i_b = i + 1;
            i_e = -1;
            in_s = true;
          }

          if (word_ok) {
            if (i_b == i_e) Dataset.perror("Wrapper.class :: empty string in --plot_labels");
            dums = all_labels.substring(i_b, i_e);
            all_strings.addElement(dums);
            i_b = i_e = -1;
            word_ok = false;
          }
        }
        i++;
      } while (i < all_labels.length());

      if (all_strings.size() <= 1)
        Dataset.perror("Wrapper.class :: <= 1 label only in --plot_labels");

      plot_labels_names = new String[all_strings.size()];
      for (i = 0; i < all_strings.size(); i++) plot_labels_names[i] = all_strings.elementAt(i);

      for (i = 0; i < all_strings.size() - 1; i++)
        for (j = i + 1; j < all_strings.size(); j++)
          if (plot_labels_names[i].equals(plot_labels_names[j]))
            Dataset.perror(
                "Wrapper.class :: same label ("
                    + plot_labels_names[i]
                    + ") repeated in --plot_labels");
    } else if (s.contains(FLAGS)) {
      dummys = ((s.substring(FLAGS.length(), s.length())).replaceAll(" ", "")).replaceAll("=", "");
      if (!dummys.substring(0, 1).equals("{"))
        Dataset.perror("Wrapper.class :: FLAGS flags does not begin with '{'");
      if (!dummys.substring(dummys.length() - 1, dummys.length()).equals("}"))
        Dataset.perror("Wrapper.class :: FLAGS flags does not end with '}'");
      dummys = (dummys.substring(1, dummys.length() - 1)).replace("\"", "");
      int b = 0, e = -1, i;
      String subs, tags, vals;
      while (e < dummys.length()) {
        b = e + 1;
        do {
          e++;
        } while ((e < dummys.length()) && (!dummys.substring(e, e + 1).equals(",")));
        subs = dummys.substring(b, e);
        if (!subs.contains(":"))
          Dataset.perror("Wrapper.class :: flags string " + subs + " not of the syntax tag:value");
        tags = subs.substring(0, subs.lastIndexOf(':'));
        vals = subs.substring(subs.lastIndexOf(':') + 1, subs.length());
        i = 0;
        do {
          if (!ALL_FLAGS[i].equals(tags)) i++;
        } while ((i < ALL_FLAGS.length) && (!ALL_FLAGS[i].equals(tags)));
        if (i == ALL_FLAGS.length)
          Dataset.perror("Wrapper.class :: flags string " + tags + " not in authorized tags");
        flags_values[i] = vals;
      }
    } else if (s.contains(IMPUTE_MISSING)) {
      impute_missing = Boolean.parseBoolean(s.substring(IMPUTE_MISSING.length(), s.length()));
    } else if (s.contains(DENSITY_ESTIMATION)) {
      density_estimation =
          Boolean.parseBoolean(s.substring(DENSITY_ESTIMATION.length(), s.length()));
      if (path_and_name_of_test_dataset == null)
        Dataset.perror("Wrapper.class :: density_estimation performed but no test set provided");
    }
  }

  public void density_estimation_save(Algorithm algo) {
    String nameFile_likelihoods =
        path_name_test + "density_estimation_likelihoods" + token_save_string + ".txt";
    String nameFile_log_likelihoods =
        path_name_test + "density_estimation_log_likelihoods" + token_save_string + ".txt";
    FileWriter f;

    int i;
    try {
      f = new FileWriter(nameFile_likelihoods);
      for (i = 0; i < algo.density_estimation_outputs_likelihoods.size(); i++) {
        f.write(
            ((int)
                    algo.density_estimation_outputs_likelihoods
                        .elementAt(i)
                        .elementAt(0)
                        .doubleValue())
                + "\t"
                + (algo.density_estimation_outputs_likelihoods
                    .elementAt(i)
                    .elementAt(1)
                    .doubleValue())
                + "\t"
                + (algo.density_estimation_outputs_likelihoods
                    .elementAt(i)
                    .elementAt(2)
                    .doubleValue())
                + "\n");
      }
      f.close();
    } catch (IOException e) {
      Dataset.perror("Experiments.class :: Saving results error in file " + nameFile_likelihoods);
    }

    try {
      f = new FileWriter(nameFile_log_likelihoods);
      for (i = 0; i < algo.density_estimation_outputs_log_likelihoods.size(); i++) {
        f.write(
            ((int)
                    algo.density_estimation_outputs_log_likelihoods
                        .elementAt(i)
                        .elementAt(0)
                        .doubleValue())
                + "\t"
                + DF6.format(
                    algo.density_estimation_outputs_log_likelihoods
                        .elementAt(i)
                        .elementAt(1)
                        .doubleValue())
                + "\t"
                + DF6.format(
                    algo.density_estimation_outputs_log_likelihoods
                        .elementAt(i)
                        .elementAt(2)
                        .doubleValue())
                + "\n");
      }
      f.close();
    } catch (IOException e) {
      Dataset.perror(
          "Experiments.class :: Saving results error in file " + nameFile_log_likelihoods);
    }
  }
}
