// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

public class Boost implements Debuggable {

  public static boolean COPYCAT_GENERATE_WITH_WHOLE_GT = true;
  // if true: re-generate the complete fake sample after each split in discriminator (copied in
  // generator)
  // otherwise, just replaces the examples of the GT-leaf whose copycat in DT has just been split in
  // the new decision tree

  public static boolean AUTHORISE_REAL_PURE_LEAVES = false;
  // if true, authorises branching with zero probability on the generator

  public static String[] KEY_NAME = {"@MatuErr"};

  public static void CHECK_NAME(String s) {
    int i = 0;
    do {
      if (KEY_NAME[i].equals(s)) return;
      i++;
    } while (i < KEY_NAME.length);
    Dataset.perror("Boost.class :: no keyword " + s);
  }

  Domain myDomain;

  int max_depth_discriminator_tree;
  int max_depth_generator_tree, max_size_discriminator_tree;

  int gaming_iters;

  double alpha;

  String name;

  String strategy_dt_grow_one_leaf;
  String strategy_game;

  public String toString() {
    String v = "";

    v += "[[" + strategy_game + "]] ";
    v += "[#GAMING_ITERS:" + gaming_iters + "|ALPHA:" + alpha + "]";

    return v;
  }

  Boost(Domain d) {
    myDomain = d;
  }

  Boost(
      Domain d,
      String nn,
      double alpha,
      String strategy_game,
      String strategy_dt_grow_one_leaf,
      int gaming_iters) {
    myDomain = d;
    name = nn;
    Boost.CHECK_NAME(name);

    this.strategy_game = strategy_game;

    max_depth_discriminator_tree = -1;
    max_depth_generator_tree = -1;

    max_size_discriminator_tree = Algorithm.IRRELEVANT_MAX_INT;

    this.strategy_dt_grow_one_leaf = strategy_dt_grow_one_leaf;
    this.gaming_iters = gaming_iters;

    this.alpha = alpha;
  }

  public Generator_Tree simple_boost(int index_in_algos) {
    return simple_boost_copycat(index_in_algos);
  }

  public Generator_Tree simple_boost_copycat(int index_in_algos) {
    // lightweight copycat run, no CV, uses all stored examples;

    boolean stop, first_init;
    int i = 0;
    String flag_dt_string;

    Vector flag_dt;
    flag_dt = null;
    String flag_gt;

    int cviter = -1;
    myDomain.myDS.compute_domain_splits(cviter);

    Generator_Tree gt = new Generator_Tree(0, this, Algorithm.MAX_DEPTH_GT);
    Generator_Node gt_node_just_split_in_dt;
    gt.init();

    Discriminator_Tree dt = new Discriminator_Tree(0, this, Algorithm.MAX_DEPTH_DT, cviter);
    dt.init_real_examples_training_leaf(null);

    myDomain.myDS.generate_examples(gt);

    stop = false;
    first_init = true;

    do {
      if (first_init) {
        dt.init();
        first_init = false;
      }
      // updates for FAKE examples

      dt.compute_train_fold_indexes_for_fakes_in_nodes_from(null, cviter);
      flag_dt = dt.one_step_grow();

      if (i % 10 == 0) System.out.print("[" + myDomain.memString() + "]");

      flag_dt_string = (String) flag_dt.elementAt(0);

      if (flag_dt_string.equals(Algorithm.DT_SPLIT_OK)) {
        gt_node_just_split_in_dt =
            gt.get_leaf_to_be_split(dt, (Discriminator_Node) flag_dt.elementAt(1));

        flag_gt =
            gt.one_step_grow_copycat(
                dt, (Discriminator_Node) flag_dt.elementAt(1), gt_node_just_split_in_dt);

        if (Boost.COPYCAT_GENERATE_WITH_WHOLE_GT) {
          myDomain.myDS.generate_examples(gt);
        } else {
          myDomain.myDS.generate_and_replace_examples(gt, gt_node_just_split_in_dt);
        }
      }
      i++;

      if ((i == gaming_iters)
          || (!flag_dt_string.equals(Algorithm.DT_SPLIT_OK))) // number of iterations
      stop = true;

    } while (!stop);
    return gt;
  }

  public double getAlpha(Discriminator_Tree dt) {
    if (alpha >= 0.0) return alpha;

    Dataset.perror("Boost.class :: No Bayes risk for alpha = " + alpha);
    return -1.0;
  }
}
