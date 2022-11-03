// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22
// class to generate data only

import java.io.*;
import java.util.*;

class FeaturePlus {
  Feature feat;
  int feat_arc_index;

  FeaturePlus(Feature f, int fai) {
    feat = f;
    feat_arc_index = fai;
  }
}

class Generate implements Debuggable {

  public static Random R = new Random();
  public static int SEPARATION_INDEX = 0;

  public static String KEY_HELP = "--help",
      KEY_GENERATOR_LINK = "-L",
      KEY_NUMBER_EXAMPLES = "-N",
      KEY_DIRECTORY = "-D",
      KEY_PREFIX = "-P",
      KEY_X = "-X",
      KEY_Y = "-Y",
      SEP = "/",
      KEY_FORCE_INTEGER_CODING = "-F",
      KEY_UNKNOWN_VALUE_CODING = "-U";
  public static String KEY_NODES = "@NODES", KEY_ARCS = "@ARCS";
  String generator_file,
      x_name,
      y_name,
      statistics_experiment,
      directory_data,
      prefix_data,
      generator_name,
      unknown_v_coding;
  boolean force_int_coding;

  int number_ex, x_index, y_index;

  Wrapper wrap;

  Generate(
      String xn,
      String yn,
      String dir,
      String pref,
      String gen_name,
      boolean force_int_coding,
      String unknown_v_coding) {
    statistics_experiment = null;
    x_name = xn;
    y_name = yn;

    directory_data = dir;
    prefix_data = pref;
    generator_name = gen_name;
    generator_file = null;

    this.force_int_coding = force_int_coding;
    if (unknown_v_coding.equals("")) this.unknown_v_coding = "-1";
    else this.unknown_v_coding = unknown_v_coding;

    wrap = null;

    x_index = y_index = -1;
  }

  public static String help() {
    String ret = "";
    ret += KEY_HELP + " : example command line\n\n";
    ret +=
        "java -Xmx10000m Generate -D Datasets/generate/ -P open_policing_hartford -U NA -F true -N"
            + " 1000 -L example-generator_open_policing_hartford.csv\n\n";
    ret += KEY_DIRECTORY + " (String) :: mandatory -- directory where to find the resource below\n";
    ret +=
        KEY_PREFIX
            + " (String) :: mandatory -- prefix of the domain (the .csv datafile must be at"
            + " Datasets/generate/open_policing_hartford.csv)\n";
    ret +=
        KEY_UNKNOWN_VALUE_CODING
            + " (String) :: optional -- representation of 'unknown' value in dataset -- default:"
            + " \"-1\"\n";
    ret +=
        KEY_FORCE_INTEGER_CODING
            + " (boolean) :: optional -- if true, recognizes integer variables and codes them as"
            + " such (otherwise, codes them as doubles) -- default: false\n";
    ret +=
        KEY_GENERATOR_LINK
            + " (String) :: mandatory -- file containing the generator to be used (the file must be"
            + " in directory Datasets/generate/open_policing_hartford/)\n";
    ret +=
        KEY_NUMBER_EXAMPLES
            + " (integer) :: optional -- #examples to be generated (the file name will be at the"
            + " same location as generator, with name ending in '_GeneratedSample.csv'; if"
            + " unspecified, just displays generator)\n";
    ret +=
        KEY_X
            + " (String) :: optional -- variable name in x value for xy density plot of generated"
            + " data\n";
    ret +=
        KEY_Y
            + " (String) :: optional -- variable name in y value for xy density plot of generated"
            + " data\n";

    return ret;
  }

  public static void main(String[] arg) {
    int i;
    String generator_n = "";
    int number_ex = -1;
    boolean kF = false;
    String kD = "", kP = "", kU = "", xn = null, yn = null;

    System.out.println("");
    System.out.println(
        "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////");
    System.out.println(
        "// Companion code to ICML'22 paper \"Generative Trees: Adversarial and Copycat\", by"
            + " Richard Nock and Mathieu Guillame-Bert //");
    System.out.println(
        "// (generating examples from a generative tree; to *train* a generative tree, try 'java"
            + " Wrapper --help')                  //");

    if (arg.length == 0) {
      System.out.println("// *No parameters*. Run 'java Generate --help' for more");
      System.exit(0);
    }

    System.out.println(
        "// Help & example run: 'java Generate --help'                                             "
            + "                                //");
    System.out.println(
        "////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////");

    for (i = 0; i < arg.length; i++) {
      if (arg[i].equals(KEY_HELP)) {
        Dataset.perror(help());
      } else if (arg[i].equals(KEY_GENERATOR_LINK)) {
        generator_n = arg[i + 1];
      } else if (arg[i].equals(KEY_DIRECTORY)) {
        kD = arg[i + 1];
      } else if (arg[i].equals(KEY_PREFIX)) {
        kP = arg[i + 1];
      } else if (arg[i].equals(KEY_UNKNOWN_VALUE_CODING)) {
        kU = arg[i + 1];
      } else if (arg[i].equals(KEY_FORCE_INTEGER_CODING)) {
        kF = Boolean.parseBoolean(arg[i + 1]);
      } else if (arg[i].equals(KEY_NUMBER_EXAMPLES)) {
        number_ex = Integer.parseInt(arg[i + 1]);
      } else if (arg[i].equals(KEY_X)) {
        xn = new String(arg[i + 1]);
      } else if (arg[i].equals(KEY_Y)) {
        yn = new String(arg[i + 1]);
      }
    }

    if (generator_n.equals(""))
      Dataset.perror("Generate.class :: no sample to generate, check parameters");

    if (kD.equals("")) Dataset.perror("Generate.class :: no directory, check parameters");

    if (kP.equals("")) Dataset.perror("Generate.class :: no prefix, check parameters");

    if (number_ex <= 0) Dataset.warning("Experiment.class :: no example generation");

    Generate ee = new Generate(xn, yn, kD, kP, generator_n, kF, kU);
    ee.go(number_ex);

    System.out.println("\nBibTex:");
    System.out.println("@inproceedings{ngbGT,");
    System.out.println("    title={Generative Trees: Adversarial and Copycat},");
    System.out.println("    author={R. Nock and M. Guillame-Bert},");
    System.out.println("    booktitle={39$^{~th}$ International Conference on Machine Learning},");
    System.out.println("    year={2022}");
    System.out.println("}\n");
  }

  public void go(int nex) {
    number_ex = nex;
    wrap = new Wrapper();
    wrap.force_integer_coding = force_int_coding;
    Unknown_Feature_Value.S_UNKNOWN = unknown_v_coding;

    wrap.path_and_name_of_domain_dataset =
        directory_data + prefix_data + SEP + prefix_data + ".csv";
    wrap.myDomain = new Domain(wrap);
    generator_file = directory_data + prefix_data + SEP + generator_name;

    System.out.println("Datafile at " + wrap.path_and_name_of_domain_dataset);
    System.out.println("Generator at " + generator_file);

    Generator_Tree gt = from_file(wrap.myDomain);
    System.out.println("GT loaded:");
    System.out.println(gt);

    if (number_ex > 0) {
      System.out.print("\nGenerating " + number_ex + " examples... ");
      Vector<Example> v_gen;

      if (((x_name != null) && (!x_name.equals(""))) && ((y_name != null) && (!y_name.equals(""))))
        v_gen = gt.generate_sample_with_density(number_ex);
      else v_gen = gt.generate_sample(number_ex);

      String save_sample_csv =
          directory_data + prefix_data + SEP + prefix_data + "_GeneratedSample.csv";
      String save_sample_density_plot_csv;

      System.out.print("ok.\nSaving generated sample in file " + save_sample_csv + " ...");
      to_file(v_gen, save_sample_csv, false, -1, -1);
      System.out.println(" ok.");

      if ((x_name != null) && (y_name != null)) {
        save_sample_density_plot_csv =
            directory_data
                + prefix_data
                + SEP
                + prefix_data
                + "_GeneratedSample_DENSITY_X_"
                + x_name
                + "_Y_"
                + y_name
                + ".csv";
        System.out.print(
            "\nSaving generated sample for density plot in file "
                + save_sample_density_plot_csv
                + " ...");
        to_file(v_gen, save_sample_density_plot_csv, true, x_index, y_index);
        System.out.println(" ok.");
      }
    }
    System.out.println("Stopping...");
    wrap.myDomain.myMemoryMonitor.stop();
  }

  public void to_file(
      Vector<Example> v_gen, String nameFile, boolean save_density, int x_index, int y_index) {
    FileWriter f;
    int i;

    try {
      f = new FileWriter(nameFile);

      if (save_density) {
        f.write(
            (wrap.myDomain.myDS.domain_feature(x_index)).name
                + Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]
                + (wrap.myDomain.myDS.domain_feature(y_index)).name
                + Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]
                + "generated_density");
      } else {
        for (i = 0; i < wrap.myDomain.myDS.number_domain_features(); i++) {
          f.write((wrap.myDomain.myDS.domain_feature(i)).name);
          if (i < wrap.myDomain.myDS.number_domain_features() - 1)
            f.write(Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
        }
      }
      f.write("\n");

      for (i = 0; i < v_gen.size(); i++)
        if (save_density)
          f.write(((Example) v_gen.elementAt(i)).toStringSaveDensity(x_index, y_index) + "\n");
        else f.write(((Example) v_gen.elementAt(i)).toStringSave() + "\n");

      f.close();
    } catch (IOException e) {
      Dataset.perror("Generate.class :: Saving results error in file " + nameFile);
    }
  }

  public Generator_Tree from_file(Domain md) {
    Generator_Tree gt = new Generator_Tree(-1, null, -1);

    boolean in_nodes = false, in_arcs = false, skip = false;

    HashSet<Generator_Node> leaves = new HashSet<>();
    Iterator lit;

    FileReader e;
    BufferedReader br;
    StringTokenizer t;

    int name,
        myParentGenerator_Node_name,
        myParentGenerator_Node_children_number,
        depth,
        i,
        j,
        index,
        feature_node_index,
        gt_depth = -1;
    double p_node;
    boolean is_leaf;
    double[] multi_p;

    Hashtable<Integer, Vector> nodes_name_to_node_and_succ_indexes_and_parent_index =
        new Hashtable<>();
    Generator_Node gn, gn_begin_node, gn_end_node, gn_parent_node;
    Integer parent_index;
    Vector vbundle, vbundle_parent_node, vbundle_begin_node, vbundle_end_node;

    Vector<Vector> arc_list_with_arcs_and_nodes_names = new Vector<>();
    Vector current_arc_and_nodes_names;
    int begin_node_name, end_node_name, feature_arc_index;
    Integer ibegin_node_name, iend_node_name, inode_name, iparent_name;

    FeaturePlus fga;
    Generator_Arc ga;
    Feature ff;
    Vector<String> modalities;
    String fname1, fname2, ftype, dum, n, dumname;
    double dmin, dmax;

    Enumeration enu;

    Vector<Integer> vsucc;
    Vector<Double> vmulti_p;

    System.out.print("Loading generator at " + generator_file + "... ");

    try {
      e = new FileReader(generator_file);
      br = new BufferedReader(e);

      while ((dum = br.readLine()) != null) {
        if ((dum.length() == 1)
            || ((dum.length() > 1)
                && (!dum.substring(0, Dataset.KEY_COMMENT.length()).equals(Dataset.KEY_COMMENT)))) {
          t = new StringTokenizer(dum, Dataset.KEY_SEPARATION_STRING[Generate.SEPARATION_INDEX]);
          n = t.nextToken();

          if (n.equals(Generate.KEY_NODES)) {
            in_nodes = true;
            in_arcs = false;
            skip = true;
          } else if (n.equals(Generate.KEY_ARCS)) {
            in_arcs = true;
            in_nodes = false;
            skip = true;
          }
          if ((!skip) && ((in_nodes) || (in_arcs))) {
            if ((in_nodes) && (in_arcs)) Dataset.perror("Generate.java :: generator file mixed up");

            if (in_nodes) {
              name = Integer.parseInt(n);
              n = t.nextToken();
              myParentGenerator_Node_name = Integer.parseInt(n);
              n = t.nextToken();
              myParentGenerator_Node_children_number = Integer.parseInt(n);
              n = t.nextToken();
              depth = Integer.parseInt(n);
              if ((gt_depth == -1) || (depth > gt_depth)) gt_depth = depth;

              n = t.nextToken();
              p_node = Double.parseDouble(n);
              n = t.nextToken();
              is_leaf = Boolean.parseBoolean(n);

              multi_p = null;
              vsucc = null;
              vmulti_p = null;
              feature_node_index = -1;
              if (!is_leaf) {
                n = t.nextToken();
                feature_node_index = Integer.parseInt(n);
                dumname = t.nextToken(); // not used

                vsucc = new Vector<>();
                vmulti_p = new Vector<>();

                while (t.hasMoreTokens()) {
                  n = t.nextToken();
                  vmulti_p.addElement(new Double(Double.parseDouble(n)));
                  n = t.nextToken();
                  vsucc.addElement(new Integer(Integer.parseInt(n)));
                }

                multi_p = new double[vmulti_p.size()];

                for (i = 0; i < multi_p.length; i++)
                  multi_p[i] = ((Double) vmulti_p.elementAt(i)).doubleValue();
              }

              gn = new Generator_Node(gt, null, myParentGenerator_Node_children_number);
              if (name == 1) gt.root = gn;

              gn.name = name;
              gn.depth = depth;
              gn.p_node = p_node;
              gn.is_leaf = is_leaf;
              gn.multi_p = multi_p;
              if (!is_leaf) {
                gn.feature_node_index = feature_node_index;
                gn.children_arcs = new Generator_Arc[multi_p.length];
              } else {
                gn.children_arcs = null;
                leaves.add(gn);
              }

              parent_index = new Integer(myParentGenerator_Node_name);

              vbundle = new Vector();
              vbundle.addElement(gn);
              vbundle.addElement(vsucc);
              vbundle.addElement(parent_index);

              nodes_name_to_node_and_succ_indexes_and_parent_index.put(new Integer(name), vbundle);
            }

            if (in_arcs) {
              begin_node_name = Integer.parseInt(n);
              n = t.nextToken();
              end_node_name = Integer.parseInt(n);
              n = t.nextToken();
              feature_arc_index = Integer.parseInt(n);
              fname1 = t.nextToken();
              fname2 = t.nextToken();
              if (!fname1.equals(fname2))
                Dataset.perror("Generate.java :: feature names " + fname1 + " != " + fname2);
              ftype = t.nextToken();

              modalities = null;
              dmin = dmax = -1.0;

              if (Feature.IS_NOMINAL(ftype)) {
                modalities = new Vector<>();
                while (t.hasMoreTokens()) modalities.addElement(new String(t.nextToken()));
              } else {
                n = t.nextToken();
                dmin = Double.parseDouble(n);
                n = t.nextToken();
                dmax = Double.parseDouble(n);
              }

              ff = new Feature(fname1, ftype, modalities, dmin, dmax, false);
              fga = new FeaturePlus(ff, feature_arc_index);

              current_arc_and_nodes_names = new Vector();
              current_arc_and_nodes_names.addElement(fga);
              current_arc_and_nodes_names.addElement(new Integer(begin_node_name));
              current_arc_and_nodes_names.addElement(new Integer(end_node_name));

              arc_list_with_arcs_and_nodes_names.addElement(current_arc_and_nodes_names);
            }
          } else if (skip) skip = false;
        }
      }

      e.close();
    } catch (IOException eee) {
      System.out.println(
          "Problem loading ." + generator_file + " resource file --- Check the access to file");
      System.exit(0);
    }

    System.out.print("ok .\nFilling nodes' parents... ");

    // fill parents
    enu = nodes_name_to_node_and_succ_indexes_and_parent_index.keys();
    while (enu.hasMoreElements()) {
      inode_name = (Integer) enu.nextElement();
      vbundle = nodes_name_to_node_and_succ_indexes_and_parent_index.get(inode_name);

      gn = (Generator_Node) vbundle.elementAt(0);
      iparent_name = (Integer) vbundle.elementAt(2);

      if (iparent_name.intValue() != -1) {
        vbundle_parent_node =
            nodes_name_to_node_and_succ_indexes_and_parent_index.get(iparent_name);
        gn_parent_node = (Generator_Node) vbundle_parent_node.elementAt(0);
        gn.myParentGenerator_Node = gn_parent_node;
      }
    }

    System.out.print("ok .\nFilling arcs... ");

    // includes arcs in nodes
    for (i = 0; i < arc_list_with_arcs_and_nodes_names.size(); i++) {
      current_arc_and_nodes_names = (Vector) arc_list_with_arcs_and_nodes_names.elementAt(i);

      fga = (FeaturePlus) current_arc_and_nodes_names.elementAt(0);
      ibegin_node_name = (Integer) current_arc_and_nodes_names.elementAt(1);
      iend_node_name = (Integer) current_arc_and_nodes_names.elementAt(2);

      vbundle_begin_node =
          nodes_name_to_node_and_succ_indexes_and_parent_index.get(ibegin_node_name);
      gn_begin_node = (Generator_Node) vbundle_begin_node.elementAt(0);
      vsucc = (Vector) vbundle_begin_node.elementAt(1);

      vbundle_end_node = nodes_name_to_node_and_succ_indexes_and_parent_index.get(iend_node_name);
      gn_end_node = (Generator_Node) vbundle_end_node.elementAt(0);

      ga = new Generator_Arc(gn_begin_node, gn_end_node, fga.feat, fga.feat_arc_index, i);
      // arc now complete

      j = 0;
      index = -1;
      do {
        if (((Integer) vsucc.elementAt(j)).equals(iend_node_name)) index = j;
        j++;
      } while ((index == -1) && (j < vsucc.size()));
      if (index == -1)
        Dataset.perror(
            "Generate.java :: index of node # " + iend_node_name + " not found at arc #" + i);

      gn_begin_node.children_arcs[index] = ga;
    }

    // finishing
    gt.leaves = leaves;
    gt.depth = gt_depth;
    gt.number_nodes = nodes_name_to_node_and_succ_indexes_and_parent_index.size();

    gt.myBoost = new Boost(wrap.myDomain);

    lit = leaves.iterator();
    while (lit.hasNext()) {
      gn = (Generator_Node) lit.next();
      gn.compute_all_features_domain();
    }

    if ((x_name != null) && (!x_name.equals(""))) {
      i = 0;
      while ((i < wrap.myDomain.myDS.number_domain_features())
          && (!(wrap.myDomain.myDS.domain_feature(i)).name.equals(x_name))) i++;
      if (!(wrap.myDomain.myDS.domain_feature(i)).name.equals(x_name))
        Dataset.perror("Generate.class :: no feature named " + x_name + " in dataset");
      x_index = i;
    }

    if ((y_name != null) && (!y_name.equals(""))) {
      i = 0;
      while ((i < wrap.myDomain.myDS.number_domain_features())
          && (!(wrap.myDomain.myDS.domain_feature(i)).name.equals(y_name))) i++;
      if (!(wrap.myDomain.myDS.domain_feature(i)).name.equals(y_name))
        Dataset.perror("Generate.class :: no feature named " + y_name + " in dataset");
      y_index = i;
    }

    System.out.print("ok.\n");

    return gt;
  }
}
