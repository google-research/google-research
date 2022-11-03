// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

class Wrapper implements Debuggable {

  public static String DATASET = "--dataset=",
      DATASET_SPEC = "--dataset_spec=",
      NUM_SAMPLES = "--num_samples=",
      WORK_DIR = "--work_dir=",
      OUTPUT_SAMPLES = "--output_samples=",
      OUTPUT_STATS = "--output_stats=",
      X = "--x=",
      Y = "--y=",
      FLAGS = "--flags",
      HELP = "--help",
      IMPUTE_MISSING = "--impute_missing=";

  public static String[] ALL_FLAGS = {
    "iterations",
    "unknown_value_coding",
    "force_integer_coding",
    "number_bins_for_histograms",
    "force_binary_coding",
    "faster_induction",
    "copycat_local_generation"
  };
  // all flag names recognized in command line in --flags = {"name" : value, ...}
  // nodes_in_gt: integer = number of nodes in the GT learned
  // unknown_value_coding: String = enforces an "unknown value" different from default
  // force_integer_coding: boolean = if true, enforce integer coding of observation variables
  // recognizable as integers ("cleaner" GT)
  // number_bins_for_histograms = integer, number of bins to compute the GT histograms of marginals
  // at the end

  public static int ALL_FLAGS_INDEX_ITERATIONS = 0,
      ALL_FLAGS_INDEX_UNKNOWN_VALUE_CODING = 1,
      ALL_FLAGS_FORCE_INTEGER_CODING = 2,
      ALL_FLAGS_NUMBER_BINS_FOR_HISTOGRAMS = 3,
      ALL_FLAGS_FORCE_BINARY_CODING = 4,
      ALL_FLAGS_FASTER_INDUCTION = 5,
      ALL_FLAGS_COPYCAT_LOCAL_GENERATION = 6;

  public static String[] DATASET_TOKENS = {
    "\"name\":", "\"path\":", "\"label\":", "\"task\":"
  }; // spec_name, spec_path, spec_label, spec_task

  public static String PREFIX_GENERATOR = "generator_";

  public String path_and_name_of_domain_dataset,
      path_to_generated_samples,
      working_directory,
      blueprint_save_name,
      spec_name,
      prefix_domain,
      spec_path,
      spec_label,
      spec_task,
      x_name,
      y_name,
      output_stats_file,
      output_stats_directory,
      generator_filename,
      densityplot_filename;
  // spec_name = prefix name

  public String[] flags_values;

  int size_generated, number_iterations; // was nums
  int index_x_name, index_y_name;

  Algorithm myAlgos;
  Domain myDomain;
  boolean force_integer_coding = false,
      force_binary_coding = true,
      faster_induction = false,
      impute_missing = false,
      has_missing_values,
      copycat_local_generation = true;
  long loading_time,
      gt_computation_time,
      marginal_computation_time,
      saving_generator_time,
      saving_stats_time,
      generate_examples_time,
      saving_generated_sample_time,
      saving_density_plot_generated_sample_time,
      imputation_time;

  Wrapper() {
    flags_values = new String[ALL_FLAGS.length];
    size_generated = number_iterations = -1;
    index_x_name = index_y_name = -1;
    densityplot_filename = null;
    x_name = y_name = null;
    path_and_name_of_domain_dataset = spec_path = null;

    loading_time =
        gt_computation_time =
            marginal_computation_time =
                saving_generator_time =
                    saving_stats_time =
                        generate_examples_time =
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
    ret += "             --num_samples=10000 \n";
    ret += "             --work_dir=${ANYDIR}/Datasets/iris/working_dir \n";
    ret +=
        "            "
            + " --output_samples=${ANYDIR}/Datasets/iris/output_samples/iris_gt_generated.csv\n";
    ret +=
        "             --output_stats=${ANYDIR}/Datasets/iris/results/generated_examples.stats \n";
    ret += "             --x=Sepal.Length --y=Sepal.Width \n";
    ret +=
        "             '--flags={\"iterations\" : \"10\", \"force_integer_coding\" : \"true\","
            + " \"force_binary_coding\" : \"true\", \"faster_induction\" : \"true\","
            + " \"unknown_value_coding\" : \"?\", \"number_bins_for_histograms\" : \"11\"}'\n";
    ret += "             --impute_missing=true\n\n";

    ret += " --dataset: path to access the.csv data file containing variable names in first line\n";
    ret += " --dataset_spec: self explanatory\n";
    ret += " --num_samples: number of generated samples\n";
    ret += " --work_dir: directory where the generator and density plots are saved\n";
    ret += " --output_samples: generated samples filename\n";
    ret +=
        " --output_stats: file to store all data related to run (execution times, GT marginals"
            + " histograms, GT tree node stats, etc)\n";
    ret +=
        " --x --y: (optional) variables used to save a 2D density plot"
            + " (x,y,denxity_value_at_(x,y))\n";
    ret += " --flags: flags...\n";
    ret +=
        "          iterations (mandatory): integer; number of splits in the GT; final number of"
            + " nodes = 2 * iteration + 1\n";
    ret +=
        "          force_integer_coding (optional): boolean; if true, recognizes integer variables"
            + " and codes them as such (otherwise, codes them as doubles) -- default: false\n";
    ret +=
        "          force_binary_coding (optional): boolean; if true, recognizes 0/1/unknown"
            + " variables and codes them as nominal, otherwise treat them as integers or doubles --"
            + " default: true\n";
    ret +=
        "          faster_induction (optional): boolean; if true, optimises training by sampling DT"
            + " splits if too many (i.e. more than "
            + Discriminator_Tree.MAX_SPLITS_BEFORE_RANDOMISATION
            + ") -- default: false\n";
    ret +=
        "          unknown_value_coding (optional): String; representation of 'unknown' value in"
            + " dataset -- default: \"-1\"\n";
    ret +=
        "          number_bins_for_histograms (optional): integer; number of bins for non-nominal"
            + " variables to store the learned GT marginals -- default: 19\n";
    ret +=
        "          copycat_local_generation (optional): boolean; if true, when copycat induction"
            + " used, after a new split in GT, example generation only replaces the affected"
            + " feature for the locally generated examples -- default: true\n";
    ret +=
        " --impute_missing: if true, uses the generated tree to impute the missing values in the"
            + " training data\n";

    return ret;
  }

  public static void main(String[] arg) {
    int i;
    Wrapper w = new Wrapper();

    System.out.println("");
    System.out.println(
        "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////");
    System.out.println(
        "// Companion code to ICML'22 paper \"Generative Trees: Adversarial and Copycat\", by"
            + " Richard Nock and Mathieu Guillame-Bert //");
    System.out.println(
        "// (copycat training of generative trees)                                                 "
            + "                                  //");

    if (arg.length == 0) {
      System.out.println("// *No parameters*. Run 'java Wrapper --help' for more");
      System.exit(0);
    }

    System.out.println(
        "// Help & example run: 'java Wrapper --help'                                              "
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

    System.out.println("\nBibTex:");
    System.out.println("@inproceedings{ngbGT,");
    System.out.println("    title={Generative Trees: Adversarial and Copycat},");
    System.out.println("    author={R. Nock and M. Guillame-Bert},");
    System.out.println("    booktitle={39$^{~th}$ International Conference on Machine Learning},");
    System.out.println("    year={2022}");
    System.out.println("}\n");
  }

  public void simple_go() {
    Algorithm.INIT();

    long b, e;
    Vector<Example> v_gen;

    System.out.print("Loading stuff start... ");
    b = System.currentTimeMillis();
    myDomain = new Domain(this);
    myAlgos = new Algorithm(myDomain);
    e = System.currentTimeMillis();
    loading_time = e - b;
    System.out.println("Loading stuff ok (time elapsed: " + loading_time + " ms).\n");

    String[] parameters = {
      "@MatuErr", "1.0", "COPYCAT", number_iterations + "", copycat_local_generation + ""
    };
    Vector<String> params = new Vector<>(Arrays.asList(parameters));
    myAlgos.addAlgorithm(params);

    System.out.print("Learning the generator... ");
    b = System.currentTimeMillis();
    Generator_Tree gt = myAlgos.simple_go();
    e = System.currentTimeMillis();
    gt_computation_time = e - b;
    System.out.println(
        "ok (time elapsed: " + gt_computation_time + " ms).\n\nGenerative tree learned: " + gt);

    System.out.print("Computing GT marginals histograms... ");
    b = System.currentTimeMillis();
    gt.compute_generator_histograms();
    e = System.currentTimeMillis();
    marginal_computation_time = e - b;
    System.out.println("ok (time elapsed: " + marginal_computation_time + " ms).");

    System.out.print("Saving the GT... ");
    b = System.currentTimeMillis();
    Generator_Tree.SAVE_GENERATOR_TREE(gt, working_directory + "/" + generator_filename, "");
    e = System.currentTimeMillis();
    saving_generator_time = e - b;
    System.out.println("ok (time elapsed: " + saving_generator_time + " ms).");

    System.out.print("Generating " + size_generated + " examples using the GT... ");
    b = System.currentTimeMillis();
    v_gen = gt.generate_sample_with_density(size_generated);
    e = System.currentTimeMillis();
    generate_examples_time = e - b;
    System.out.println("ok (time elapsed: " + generate_examples_time + " ms).");

    if ((has_missing_values) && (impute_missing)) {
      System.out.print("Imputing examples using the GT... ");
      b = System.currentTimeMillis();
      impute_and_save(gt);
      e = System.currentTimeMillis();
      imputation_time = e - b;
      System.out.println("ok (time elapsed: " + imputation_time + " ms).");
    }

    System.out.print("Saving generated sample... ");
    b = System.currentTimeMillis();
    save_sample(v_gen);
    e = System.currentTimeMillis();
    saving_generated_sample_time = e - b;
    System.out.println("ok (time elapsed: " + saving_generated_sample_time + " ms).");

    if ((x_name != null) && (y_name != null)) {
      System.out.print("Saving density plot sample... ");
      b = System.currentTimeMillis();
      save_density_plot_sample(v_gen, index_x_name, index_y_name);
      e = System.currentTimeMillis();
      saving_density_plot_generated_sample_time = e - b;
      System.out.println(
          "ok (time elapsed: " + saving_density_plot_generated_sample_time + " ms).");
    }

    System.out.print("Saving stats file... ");
    b = System.currentTimeMillis();
    save_stats(gt);
    e = System.currentTimeMillis();
    saving_stats_time = e - b;
    System.out.println("ok (time elapsed: " + saving_stats_time + " ms).");

    System.out.println("All finished. Stopping...");
    myDomain.myMemoryMonitor.stop();
  }

  public void save_density_plot_sample(Vector<Example> v_gen, int x, int y) {
    FileWriter f;
    int i;

    String nameFile = working_directory + "/" + densityplot_filename;

    try {
      f = new FileWriter(nameFile);

      f.write("//" + x_name + "," + y_name + ",density_value\n");

      for (i = 0; i < v_gen.size(); i++)
        f.write(
            ((Example) v_gen.elementAt(i)).toStringSaveDensity(index_x_name, index_y_name) + "\n");

      f.close();
    } catch (IOException e) {
      Dataset.perror("Experiments.class :: Saving results error in file " + nameFile);
    }
  }

  public void save_sample(Vector<Example> v_gen) {
    FileWriter f;
    int i;

    String nameFile = path_to_generated_samples + "/" + blueprint_save_name;

    try {
      f = new FileWriter(nameFile);

      for (i = 0; i < myDomain.myDS.features_names_from_file.length; i++) {
        f.write(myDomain.myDS.features_names_from_file[i]);
        if (i < myDomain.myDS.features_names_from_file.length - 1)
          f.write(Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
      }
      f.write("\n");

      for (i = 0; i < v_gen.size(); i++)
        f.write(((Example) v_gen.elementAt(i)).toStringSave() + "\n");

      f.close();
    } catch (IOException e) {
      Dataset.perror("Experiments.class :: Saving results error in file " + nameFile);
    }
  }

  public void impute_and_save(Generator_Tree gt) {
    FileWriter f;
    int i, j;

    String nameFile = working_directory + "/" + spec_name + "_imputed.csv";
    Example ee, ee_cop;

    try {
      f = new FileWriter(nameFile);

      for (i = 0; i < myDomain.myDS.features_names_from_file.length; i++) {
        f.write(myDomain.myDS.features_names_from_file[i]);
        if (i < myDomain.myDS.features_names_from_file.length - 1)
          f.write(Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
      }
      f.write("\n");

      for (i = 0; i < myDomain.myDS.number_examples_total_from_file; i++) {
        if (i % (myDomain.myDS.number_examples_total_from_file / 10) == 0)
          System.out.print((i / (myDomain.myDS.number_examples_total_from_file / 10)) * 10 + "% ");

        ee = (Example) myDomain.myDS.examples_from_file.elementAt(i);
        ee_cop = Example.copyOf(ee);

        if (ee.contains_unknown_values()) {
          gt.impute_all_values_from_one_leaf(ee_cop);
        }
        f.write(ee_cop.toStringSave() + "\n");
      }
      f.close();
    } catch (IOException e) {
      Dataset.perror("Experiments.class :: Saving results error in file " + nameFile);
    }
  }

  public void save_stats(Generator_Tree gt) {
    int i;
    FileWriter f = null;
    int[] node_counts = new int[myDomain.myDS.features_names_from_file.length];
    String output_stats_additional = output_stats_file + "_more.txt";

    long time_gt_plus_generation = gt_computation_time + generate_examples_time;
    long time_gt_plus_imputation = -1;
    if ((has_missing_values) && (impute_missing))
      time_gt_plus_imputation = gt_computation_time + imputation_time;

    try {
      f = new FileWriter(output_stats_file);

      f.write("{\n");

      f.write(
          " \"running_time_seconds\": "
              + DF8.format(((double) gt_computation_time + generate_examples_time) / 1000)
              + ",\n");
      f.write(" \"gt_number_nodes\": " + gt.number_nodes + ",\n");
      f.write(" \"gt_depth\": " + gt.depth + ",\n");
      f.write(
          " \"running_time_gt_training_plus_exemple_generation\": "
              + DF8.format(((double) time_gt_plus_generation / 1000))
              + ",\n");
      if ((has_missing_values) && (impute_missing))
        f.write(
            " \"running_time_gt_training_plus_imputation\": "
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
      f.write("// Time to generate sample: " + generate_examples_time + " ms.\n");

      // saves the gt histograms for comparison
      f.write("\n// GT Histograms of marginals\n");
      for (i = 0; i < gt.gt_histograms.size(); i++) {
        f.write((gt.gt_histograms.elementAt(i)).toStringSave());
        f.write("//\n");
      }

      gt.root.recursive_fill_node_counts(node_counts, myDomain.myDS.features_names_from_file);
      f.write("\n// GT node counts per feature name\n");
      for (i = 0; i < myDomain.myDS.features_names_from_file.length; i++) {
        f.write("// " + myDomain.myDS.features_names_from_file[i] + " : " + node_counts[i] + "\n");
      }

      f.close();
    } catch (IOException e) {
      Dataset.perror("Wrapper.class :: Saving results error in file " + output_stats_additional);
    }
  }

  public void summary() {
    int i;
    System.out.println("\nRunning copycat generator training with the following inputs:");
    System.out.println(" * dataset path to train generator:" + path_and_name_of_domain_dataset);
    System.out.println(" * working directory:" + working_directory);
    System.out.println(
        " * generated samples ("
            + size_generated
            + " examples) stored in directory "
            + path_to_generated_samples
            + " with filename "
            + blueprint_save_name);
    System.out.println(
        " * generator (gt) stored in working directory with filename " + generator_filename);
    System.out.println(" * stats file at " + output_stats_file);

    if (spec_path == null) Dataset.warning(" No path information in --dataset_spec");
    else if (!path_and_name_of_domain_dataset.equals(spec_path))
      Dataset.warning(" Non identical information in --dataset_spec path vs --dataset\n");

    if (impute_missing)
      System.out.println(
          " * imputed sample saved at filename "
              + working_directory
              + "/"
              + spec_name
              + "_imputed.csv");

    if ((x_name != null) && (y_name != null) && (!x_name.equals("")) && (!y_name.equals(""))) {
      densityplot_filename =
          blueprint_save_name.substring(0, blueprint_save_name.lastIndexOf('.'))
              + "_2DDensity_plot_X_"
              + x_name
              + "_Y_"
              + y_name
              + blueprint_save_name.substring(
                  blueprint_save_name.lastIndexOf('.'), blueprint_save_name.length());
      System.out.println(
          " * 2D density plot for ("
              + x_name
              + ","
              + y_name
              + ") stored in working directory with filename "
              + densityplot_filename);
    }
    System.out.print(" * flags (non null): ");
    for (i = 0; i < flags_values.length; i++)
      if (flags_values[i] != null)
        System.out.print("[" + ALL_FLAGS[i] + ":" + flags_values[i] + "] ");
    System.out.println("");

    if (flags_values[ALL_FLAGS_INDEX_UNKNOWN_VALUE_CODING] != null)
      Unknown_Feature_Value.S_UNKNOWN = flags_values[ALL_FLAGS_INDEX_UNKNOWN_VALUE_CODING];

    if (flags_values[ALL_FLAGS_FORCE_INTEGER_CODING] != null)
      force_integer_coding = Boolean.parseBoolean(flags_values[ALL_FLAGS_FORCE_INTEGER_CODING]);

    if (flags_values[ALL_FLAGS_FORCE_BINARY_CODING] != null)
      force_binary_coding = Boolean.parseBoolean(flags_values[ALL_FLAGS_FORCE_BINARY_CODING]);

    if (flags_values[ALL_FLAGS_FASTER_INDUCTION] != null)
      faster_induction = Boolean.parseBoolean(flags_values[ALL_FLAGS_FASTER_INDUCTION]);

    if (flags_values[ALL_FLAGS_INDEX_ITERATIONS] != null)
      number_iterations = Integer.parseInt(flags_values[ALL_FLAGS_INDEX_ITERATIONS]);

    if (flags_values[ALL_FLAGS_COPYCAT_LOCAL_GENERATION] != null)
      copycat_local_generation =
          Boolean.parseBoolean(flags_values[ALL_FLAGS_COPYCAT_LOCAL_GENERATION]);
    Boost.COPYCAT_GENERATE_WITH_WHOLE_GT = !copycat_local_generation;

    if (flags_values[ALL_FLAGS_NUMBER_BINS_FOR_HISTOGRAMS] != null) {
      Histogram.NUMBER_CONTINUOUS_FEATURE_BINS =
          Integer.parseInt(flags_values[ALL_FLAGS_NUMBER_BINS_FOR_HISTOGRAMS]);
      Histogram.MAX_NUMBER_INTEGER_FEATURE_BINS =
          Integer.parseInt(flags_values[ALL_FLAGS_NUMBER_BINS_FOR_HISTOGRAMS]);
    }

    Dataset.NUMBER_GENERATED_EXAMPLES_DEFAULT = -1;
    Discriminator_Tree.USE_OBSERVED_FEATURE_VALUES_FOR_SPLITS = true;
    Discriminator_Tree.RANDOMISE_SPLIT_FINDING_WHEN_TOO_MANY_SPLITS = faster_induction;

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
    } else if (s.contains(X)) {
      x_name = s.substring(X.length(), s.length());
    } else if (s.contains(Y)) {
      y_name = s.substring(Y.length(), s.length());
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
    }
    if ((x_name != null) && (y_name != null) && (x_name.equals(y_name)))
      Dataset.perror(
          "Wrapper.class :: density plot requested on the same X and Y variable = " + x_name);
  }
}
