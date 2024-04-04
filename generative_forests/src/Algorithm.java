// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Algorithm
 *****/

class SplitDetails implements Debuggable {
  // records the split details of a split
  // important for split_observations, since some observations can have unknown values and thus this
  // needs to be computed & recorded ONCE

  int index_feature;
  int index_split;

  int[][] split_observations;

  SplitDetails(int index_f, int index_s, int[][] split_o) {
    index_feature = index_f;
    index_split = index_s;
    split_observations = split_o;
  }
}

public class Algorithm implements Debuggable {
  public static String
      LEAF_CHOICE_RANDOMIZED_WITH_NON_ZERO_EMPIRICAL_MEASURE =
          "LEAF_CHOICE_RANDOMIZED_WITH_NON_ZERO_EMPIRICAL_MEASURE",
      LEAF_CHOICE_HEAVIEST_LEAF_AMONG_TREES = "LEAF_CHOICE_HEAVIEST_LEAF_AMONG_TREES";
  // different ways of picking a leaf

  public static String[] LEAF_CHOICE = {
    LEAF_CHOICE_RANDOMIZED_WITH_NON_ZERO_EMPIRICAL_MEASURE, LEAF_CHOICE_HEAVIEST_LEAF_AMONG_TREES
  };

  public static double MINIMUM_EMPIRICAL_CARD_AT_NEW_LEAF_FOR_BOOSTING = 1.0;
  // boosting can "peel off" support with zero empirical card: this requires minimum local card at
  // new leaf

  public static double MINIMUM_P_R_FOR_BOOSTING = EPS2;
  // minimum estimated weight of empirical measure to allow a split for EOGT

  public static int MAXIMAL_NUMBER_OF_SPLIT_TESTS_TRIES_PER_BOOSTING_ITERATION = 1000;
  // if the total number of tests to be done is larger than this, does random picking of tests for
  // this number of iterations (else is exhaustive)

  public static int STEPS_TO_COMPUTE_DENSITY_ESTIMATION = 100;

  // interval to save density estimation values

  public static void CHECK_LEAF_CHOICE_VALID(String s) {
    int i = 0;
    do {
      if (LEAF_CHOICE[i].equals(s)) return;
      else i++;
    } while (i < LEAF_CHOICE.length);
    Dataset.perror("Algorithm.class :: no such choice as " + s + " for leaf choice");
  }

  Dataset myDS;
  GenerativeModelBasedOnEnsembleOfTrees geot;

  // parameters for learning
  int nb_iterations, initial_number_of_trees;
  boolean nb_trees_increase_allowed;
  String how_to_choose_leaf, splitting_method;

  public static String[] DENSITY_ESTIMATION_OUTPUT_TYPE = {"LIKELIHOODS", "LOG_LIKELIHOODS"};
  public static int INDEX_DENSITY_ESTIMATION_OUTPUT_TYPE = 0;
  // type of output for density estimation

  Vector<Vector<Double>> density_estimation_outputs_likelihoods;
  Vector<Vector<Double>> density_estimation_outputs_log_likelihoods;

  Algorithm(
      Dataset ds,
      int name,
      int nb_iterations,
      int initial_number_of_trees,
      String split_m,
      boolean generative_forest) {
    this(ds, name);

    if (generative_forest) geot.generative_forest();
    else geot.ensemble_of_generative_trees();

    this.nb_iterations = nb_iterations;
    this.initial_number_of_trees = initial_number_of_trees;

    this.how_to_choose_leaf = Algorithm.LEAF_CHOICE_HEAVIEST_LEAF_AMONG_TREES;
    splitting_method = split_m;

    nb_trees_increase_allowed = false;
  }

  Algorithm(Dataset ds, int n) {
    myDS = ds;
    geot = new GenerativeModelBasedOnEnsembleOfTrees(ds, n);
  }

  public GenerativeModelBasedOnEnsembleOfTrees learn_geot() {
    if (nb_trees_increase_allowed) Dataset.perror("NOT YET");

    if (geot.generative_forest) System.out.print(" [GF]  ");
    else if (geot.ensemble_of_generative_trees) System.out.print("[EOGT] ");

    geot.init(initial_number_of_trees);

    HashSet<MeasuredSupportAtTupleOfNodes> tuples_of_leaves_with_positive_measure_geot =
        new HashSet<>();

    if (splitting_method.equals(Wrapper.BOOSTING))
      tuples_of_leaves_with_positive_measure_geot.add(
          new MeasuredSupportAtTupleOfNodes(geot, false, true, -1));

    int i;
    Node leaf;
    boolean ok;

    Vector<Double> current_likelihood;
    Vector<Double> current_log_likelihood;

    double[][] avs = new double[2][]; // [0][] = likelihood; [1][] = loglikelihood
    avs[0] = new double[2];
    avs[1] = new double[2];

    if ((geot.generative_forest) && (myDS.myDomain.myW.density_estimation)) {
      density_estimation_outputs_likelihoods = new Vector<>();
      density_estimation_outputs_log_likelihoods = new Vector<>();
    }

    for (i = 0; i < nb_iterations; i++) {

      leaf = null;
      ok = true;

      leaf = choose_leaf(how_to_choose_leaf);

      if (leaf == null) {
        Dataset.warning(" no splittable leaf found ");
        ok = false;
      }

      if (ok) {
        if (geot.generative_forest)
          ok = split_generative_forest(leaf, tuples_of_leaves_with_positive_measure_geot);
        else if (geot.ensemble_of_generative_trees)
          ok =
              split_ensemble_of_generative_trees(leaf, tuples_of_leaves_with_positive_measure_geot);
        else Dataset.perror("Algorithm.class :: no such option available yet");
      }

      if (!ok) {
        System.out.print("X ");
        break;
      } else System.out.print(".");

      if (i % 10 == 0) System.out.print(myDS.myDomain.memString());

      leaf.myTree.checkConsistency();

      if ((i % Algorithm.STEPS_TO_COMPUTE_DENSITY_ESTIMATION == 0)
          && (geot.generative_forest)
          && (myDS.myDomain.myW.density_estimation)) {
        current_likelihood = new Vector<>();
        current_log_likelihood = new Vector<>();

        avs = geot.expected_density_estimation_output();

        current_likelihood.addElement(new Double(i));
        current_likelihood.addElement(new Double(avs[0][0]));
        current_likelihood.addElement(new Double(avs[0][1]));
        density_estimation_outputs_likelihoods.addElement(current_likelihood);

        current_log_likelihood.addElement(new Double(i));
        current_log_likelihood.addElement(new Double(avs[1][0]));
        current_log_likelihood.addElement(new Double(avs[1][1]));
        density_estimation_outputs_log_likelihoods.addElement(current_log_likelihood);
      }
    }

    System.out.print(" [soft_probs] ");
    geot.compute_probabilities_soft();

    return geot;
  }

  Node choose_leaf(String how_to_choose_leaf) {
    Algorithm.CHECK_LEAF_CHOICE_VALID(how_to_choose_leaf);
    // a leaf has a handle to its tree so can pick a leaf regardless of the tree
    if (geot.trees == null) Dataset.perror("Algorithm.class :: no tree to choose a leaf from");

    int i, j, meas = -1;
    Node n, ret = null;
    boolean firstleaf = true;
    Iterator it;
    if (how_to_choose_leaf.equals(Algorithm.LEAF_CHOICE_HEAVIEST_LEAF_AMONG_TREES)) {
      for (i = 0; i < geot.trees.size(); i++) {
        it = geot.trees.elementAt(i).leaves.iterator();
        while (it.hasNext()) {
          n = (Node) it.next();
          if ((n.observations_in_node)
              && (n.has_feature_tests())
              && ((firstleaf) || (n.observation_indexes_in_node.length > meas))) {
            firstleaf = false;
            ret = n;
            meas = n.observation_indexes_in_node.length;
          }
        }
      }
    }

    return ret;
  }

  public boolean split_generative_forest(Node leaf, HashSet<MeasuredSupportAtTupleOfNodes> tol) {
    if (!leaf.is_leaf) Dataset.perror("Algorithm.class :: " + leaf + " is not a leaf");

    FeatureTest[][] all_feature_tests = new FeatureTest[leaf.node_support.dim()][];
    // x = feature index, y = split index
    boolean[] splittable_feature = new boolean[leaf.node_support.dim()];
    boolean at_least_one_splittable_feature = false;

    SplitDetails sd = null;

    int i, nb_total_splits = 0, j, index_f = -1, index_s = -1;
    Vector<FeatureTest> dumft;

    for (i = 0; i < leaf.node_support.dim(); i++) {
      dumft = FeatureTest.ALL_FEATURE_TESTS(leaf.node_support.feature(i), myDS);
      if (dumft != null) {
        at_least_one_splittable_feature = true;
        splittable_feature[i] = true;
        all_feature_tests[i] = new FeatureTest[dumft.size()];
        for (j = 0; j < dumft.size(); j++) all_feature_tests[i][j] = dumft.elementAt(j);
        nb_total_splits += all_feature_tests[i].length;
      } else {
        splittable_feature[i] = false;
        all_feature_tests[i] = new FeatureTest[0];
      }
    }

    if (!at_least_one_splittable_feature)
      Dataset.perror("Algorithm.class :: no splittable feature for node " + leaf);

    if (splitting_method.equals(Wrapper.BOOSTING)) {
      if (Wrapper.FAST_SPLITTING)
        sd =
            split_top_down_boosting_generative_forest_fast(
                leaf, splittable_feature, all_feature_tests, tol);
      else
        sd =
            split_top_down_boosting_generative_forest(
                leaf, splittable_feature, all_feature_tests, tol);
    } else Dataset.perror("Algorithm.class :: no such split choice as " + splitting_method);

    if (sd == null) return false;

    index_f = sd.index_feature;
    index_s = sd.index_split;
    int[][] new_split_observations = sd.split_observations;

    // feature found at index index_f, split_index to be used at all_feature_tests[index_f][index_s]
    // node update

    if ((leaf.is_leaf)
        || (leaf.node_feature_test != all_feature_tests[index_f][index_s])
        || (leaf.node_feature_split_index != index_f))
      Dataset.perror(
          "Algorithm.class :: leaf should have been updated ("
              + leaf.is_leaf
              + ") ("
              + leaf.node_feature_test
              + " vs "
              + all_feature_tests[index_f][index_s]
              + ") ("
              + leaf.node_feature_split_index
              + " vs "
              + index_s
              + ")");

    // create two new nodes

    Feature[] new_features =
        leaf.node_feature_test.split_feature(
            myDS, leaf.node_support.feature(leaf.node_feature_split_index), true, true);
    Support[] new_supports =
        leaf.node_support.split_support(leaf.node_feature_split_index, new_features);

    Node new_left_leaf =
        new Node(
            leaf.myTree,
            leaf.myTree.number_nodes,
            leaf.depth + 1,
            new_split_observations[0],
            new_supports[0]);
    Node new_right_leaf =
        new Node(
            leaf.myTree,
            leaf.myTree.number_nodes + 1,
            leaf.depth + 1,
            new_split_observations[1],
            new_supports[1]);

    if (new_split_observations[0].length + new_split_observations[1].length
        != leaf.observation_indexes_in_node.length)
      Dataset.perror("Algorithm.class :: error in splitting supports, mismatch in #examples");

    leaf.p_left =
        (double) new_split_observations[0].length
            / ((double) leaf.observation_indexes_in_node.length);
    leaf.p_right =
        (double) new_split_observations[1].length
            / ((double) leaf.observation_indexes_in_node.length);

    if (splitting_method.equals(Wrapper.BOOSTING)) {
      // unmarking the elements in tol
      Iterator it = tol.iterator();
      MeasuredSupportAtTupleOfNodes ms;

      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();
        if (ms.marked_for_update) ms.unmark_for_update(new_left_leaf, new_right_leaf);
      }
    }

    // update tree's statistics

    leaf.myTree.statistics_number_of_nodes_for_each_feature[index_f]++;

    // link nodes

    leaf.left_child = new_left_leaf;
    leaf.right_child = new_right_leaf;

    leaf.left_child.p_reach = leaf.p_reach * leaf.p_left;
    leaf.right_child.p_reach = leaf.p_reach * leaf.p_right;

    // update tree leaves & size

    leaf.myTree.number_nodes += 2;
    boolean test_remove =
        leaf.myTree.remove_leaf_from_tree_structure(leaf); // leaf.myTree.leaves.remove(leaf);
    if (!test_remove)
      Dataset.perror("Algorithm.class :: Leaf " + leaf.name + " not found in tree's leaves");

    if (leaf.depth > leaf.myTree.depth)
      Dataset.perror(
          "Algorithm.class :: Leaf "
              + leaf.name
              + " at depth "
              + leaf.depth
              + " > leaf.myTree depth = "
              + leaf.myTree.depth);
    if (leaf.depth == leaf.myTree.depth) leaf.myTree.depth++;

    leaf.myTree.add_leaf_to_tree_structure(new_left_leaf);
    leaf.myTree.add_leaf_to_tree_structure(new_right_leaf);

    return true;
  }

  public boolean split_ensemble_of_generative_trees(
      Node leaf, HashSet<MeasuredSupportAtTupleOfNodes> tol) {
    if (!leaf.is_leaf) Dataset.perror("Algorithm.class :: " + leaf + " is not a leaf");

    FeatureTest[][] all_feature_tests = new FeatureTest[leaf.node_support.dim()][];
    // x = feature index, y = split index
    boolean[] splittable_feature = new boolean[leaf.node_support.dim()];
    boolean at_least_one_splittable_feature = false;

    SplitDetails sd = null;

    int i, nb_total_splits = 0, j, index_f = -1, index_s = -1;
    Vector<FeatureTest> dumft;

    for (i = 0; i < leaf.node_support.dim(); i++) {
      dumft = FeatureTest.ALL_FEATURE_TESTS(leaf.node_support.feature(i), myDS);
      if (dumft != null) {
        at_least_one_splittable_feature = true;
        splittable_feature[i] = true;
        all_feature_tests[i] = new FeatureTest[dumft.size()];
        for (j = 0; j < dumft.size(); j++) all_feature_tests[i][j] = dumft.elementAt(j);
        nb_total_splits += all_feature_tests[i].length;
      } else {
        splittable_feature[i] = false;
        all_feature_tests[i] = new FeatureTest[0];
      }
    }

    if (!at_least_one_splittable_feature)
      Dataset.perror("Algorithm.class :: no splittable feature for node " + leaf);

    if (splitting_method.equals(Wrapper.BOOSTING)) {
      sd =
          split_top_down_boosting_ensemble_of_generative_trees_fast(
              leaf, splittable_feature, all_feature_tests, tol);
    } else Dataset.perror("Algorithm.class :: no such split choice as " + splitting_method);

    if (sd == null) return false;

    index_f = sd.index_feature;
    index_s = sd.index_split;
    int[][] new_split_observations = sd.split_observations;

    // feature found at index index_f, split_index to be used at all_feature_tests[index_f][index_s]
    // node update

    if ((leaf.is_leaf)
        || (leaf.node_feature_test != all_feature_tests[index_f][index_s])
        || (leaf.node_feature_split_index != index_f))
      Dataset.perror(
          "Algorithm.class :: leaf should have been updated ("
              + leaf.is_leaf
              + ") ("
              + leaf.node_feature_test
              + " vs "
              + all_feature_tests[index_f][index_s]
              + ") ("
              + leaf.node_feature_split_index
              + " vs "
              + index_s
              + ")");

    // create two new nodes

    Feature[] new_features =
        leaf.node_feature_test.split_feature(
            myDS, leaf.node_support.feature(leaf.node_feature_split_index), true, true);
    Support[] new_supports =
        leaf.node_support.split_support(leaf.node_feature_split_index, new_features);

    Node new_left_leaf =
        new Node(
            leaf.myTree,
            leaf.myTree.number_nodes,
            leaf.depth + 1,
            new_split_observations[0],
            new_supports[0]);
    Node new_right_leaf =
        new Node(
            leaf.myTree,
            leaf.myTree.number_nodes + 1,
            leaf.depth + 1,
            new_split_observations[1],
            new_supports[1]);

    if (new_split_observations[0].length + new_split_observations[1].length
        != leaf.observation_indexes_in_node.length)
      Dataset.perror("Algorithm.class :: error in splitting supports, mismatch in #examples");

    leaf.p_left =
        (double) new_split_observations[0].length
            / ((double) leaf.observation_indexes_in_node.length);
    leaf.p_right =
        (double) new_split_observations[1].length
            / ((double) leaf.observation_indexes_in_node.length);

    if (splitting_method.equals(Wrapper.BOOSTING)) {
      // unmarking the elements in tol
      Iterator it = tol.iterator();
      MeasuredSupportAtTupleOfNodes ms;

      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();
        if (ms.marked_for_update) ms.unmark_for_update(new_left_leaf, new_right_leaf);
      }
    }

    // update tree's statistics

    leaf.myTree.statistics_number_of_nodes_for_each_feature[index_f]++;

    // link nodes

    leaf.left_child = new_left_leaf;
    leaf.right_child = new_right_leaf;

    leaf.left_child.p_reach = leaf.p_reach * leaf.p_left;
    leaf.right_child.p_reach = leaf.p_reach * leaf.p_right;

    // update tree leaves & size

    leaf.myTree.number_nodes += 2;
    boolean test_remove =
        leaf.myTree.remove_leaf_from_tree_structure(leaf); // leaf.myTree.leaves.remove(leaf);
    if (!test_remove)
      Dataset.perror("Algorithm.class :: Leaf " + leaf.name + " not found in tree's leaves");

    if (leaf.depth > leaf.myTree.depth)
      Dataset.perror(
          "Algorithm.class :: Leaf "
              + leaf.name
              + " at depth "
              + leaf.depth
              + " > leaf.myTree depth = "
              + leaf.myTree.depth);
    if (leaf.depth == leaf.myTree.depth) leaf.myTree.depth++;

    leaf.myTree.add_leaf_to_tree_structure(new_left_leaf);
    leaf.myTree.add_leaf_to_tree_structure(new_right_leaf);

    return true;
  }

  public HashSet<String> filter_for_leaf(
      Hashtable<String, MeasuredSupportAtTupleOfNodes> all_measured_supports_at_intersection,
      Node leaf) {
    // returns a subset of keys of all_measured_supports_at_intersection having leaf in them
    HashSet<String> ret = new HashSet<>();
    Vector<Node> all_nodes;
    boolean tree_done;

    int i;
    for (i = 0; i < geot.trees.size(); i++) {
      all_nodes = new Vector<>();
      tree_done = false;
    }

    return ret;
  }

  public SplitDetails split_top_down_boosting_generative_forest(
      Node leaf,
      boolean[] splittable_feature,
      FeatureTest[][] all_feature_tests,
      HashSet<MeasuredSupportAtTupleOfNodes> tol) {
    HashSet<MeasuredSupportAtTupleOfNodes> subset_with_leaf = new HashSet<>();
    HashSet<MeasuredSupportAtTupleOfNodes> new_candidates_after_split;
    HashSet<MeasuredSupportAtTupleOfNodes> best_candidates_for_left_after_split = null;
    HashSet<MeasuredSupportAtTupleOfNodes> best_candidates_for_right_after_split = null;
    Iterator it = tol.iterator();
    MeasuredSupportAtTupleOfNodes ms;

    int number_obs_tot = 0,
        i,
        j,
        k,
        index_f = -1,
        index_s = -1,
        number_left_obs,
        number_right_obs,
        iter = 0,
        index_try;
    double cur_Bayes, p_ccstar, opt_Bayes = -1.0;
    boolean found, at_least_one;

    while (it.hasNext()) {
      ms = (MeasuredSupportAtTupleOfNodes) it.next();
      if (ms.tree_nodes_support_contains_node(leaf)) {
        subset_with_leaf.add(ms);
        number_obs_tot += ms.generative_support.local_empirical_measure.observations_indexes.length;
      }
    }

    if (number_obs_tot != leaf.observation_indexes_in_node.length)
      Dataset.perror(
          "Algorithm.class :: mismatch between the total #observations ("
              + number_obs_tot
              + ") and leaf weight ("
              + leaf.observation_indexes_in_node.length
              + ")");

    if (subset_with_leaf.size() == 0)
      Dataset.perror(
          "Algorithm.class :: no subsets of MeasuredSupportAtTupleOfNodes with leaf " + leaf);

    // speed up
    Vector<Integer[]> couple_feature_index_feature_test_index = new Vector<>();

    for (i = 0; i < splittable_feature.length; i++)
      if (splittable_feature[i])
        for (j = 0; j < all_feature_tests[i].length; j++)
          couple_feature_index_feature_test_index.addElement(
              new Integer[] {new Integer(i), new Integer(j)});

    List<Integer> shuffled_indexes = new ArrayList<>();
    for (i = 0; i < couple_feature_index_feature_test_index.size(); i++) shuffled_indexes.add(i);
    Collections.shuffle(shuffled_indexes);

    int
        max_number_tries =
            (couple_feature_index_feature_test_index.size()
                    > Algorithm.MAXIMAL_NUMBER_OF_SPLIT_TESTS_TRIES_PER_BOOSTING_ITERATION)
                ? Algorithm.MAXIMAL_NUMBER_OF_SPLIT_TESTS_TRIES_PER_BOOSTING_ITERATION
                : couple_feature_index_feature_test_index.size(),
        dumi;

    // works regardless of the option chosen

    found = false;
    for (k = 0; k < max_number_tries; k++) {
      index_try = shuffled_indexes.get(shuffled_indexes.size() - 1).intValue();
      i = couple_feature_index_feature_test_index.elementAt(index_try)[0].intValue();
      j = couple_feature_index_feature_test_index.elementAt(index_try)[1].intValue();

      dumi = shuffled_indexes.remove(shuffled_indexes.size() - 1);

      iter++;

      new_candidates_after_split = new HashSet<>();
      cur_Bayes = 0.0;

      // computes all split_top_down_boosting_statistics & current Bayes entropy (HL in (7))
      number_left_obs = number_right_obs = 0;
      it = subset_with_leaf.iterator();
      at_least_one = false;
      while (it.hasNext()) {
        at_least_one = true;
        ms = (MeasuredSupportAtTupleOfNodes) it.next();
        ms.split_top_down_boosting_statistics =
            ms.split_for_boosting_computations(all_feature_tests[i][j], i);

        if ((ms.split_top_down_boosting_statistics[0] != null)
            && (ms.split_top_down_boosting_statistics[0]
                    .generative_support
                    .local_empirical_measure
                    .observations_indexes
                != null))
          number_left_obs +=
              ms.split_top_down_boosting_statistics[0]
                  .generative_support
                  .local_empirical_measure
                  .observations_indexes
                  .length;

        if ((ms.split_top_down_boosting_statistics[1] != null)
            && (ms.split_top_down_boosting_statistics[1]
                    .generative_support
                    .local_empirical_measure
                    .observations_indexes
                != null))
          number_right_obs +=
              ms.split_top_down_boosting_statistics[1]
                  .generative_support
                  .local_empirical_measure
                  .observations_indexes
                  .length;

        p_ccstar =
            (Statistics.PRIOR
                    * ((double)
                        ms.generative_support.local_empirical_measure.observations_indexes.length)
                    / ((double) myDS.observations_from_file.size()))
                + ((1.0 - Statistics.PRIOR)
                    * ms.generative_support.support.weight_uniform_distribution);

        cur_Bayes +=
            (p_ccstar
                * Statistics.MU_BAYES_GENERATIVE_FOREST(
                    ms.split_top_down_boosting_statistics, 1.0));
      }

      // Bayes entropy improvement ?
      if ((at_least_one)
          && (number_left_obs >= Algorithm.MINIMUM_EMPIRICAL_CARD_AT_NEW_LEAF_FOR_BOOSTING)
          && (number_right_obs >= Algorithm.MINIMUM_EMPIRICAL_CARD_AT_NEW_LEAF_FOR_BOOSTING)
          && ((!found) || (cur_Bayes < opt_Bayes))) {
        index_f = i;
        index_s = j;
        opt_Bayes = cur_Bayes;
        found = true;

        best_candidates_for_left_after_split = new HashSet<>();
        best_candidates_for_right_after_split = new HashSet<>();
        it = subset_with_leaf.iterator();
        while (it.hasNext()) {
          ms = (MeasuredSupportAtTupleOfNodes) it.next();
          if (ms.split_top_down_boosting_statistics[0] != null)
            best_candidates_for_left_after_split.add(ms.split_top_down_boosting_statistics[0]);
          if (ms.split_top_down_boosting_statistics[1] != null)
            best_candidates_for_right_after_split.add(ms.split_top_down_boosting_statistics[1]);
        }
      }
    }

    // update leaf
    if ((index_f == -1) || (index_s == -1)) return null;

    leaf.node_feature_test = all_feature_tests[index_f][index_s];
    leaf.node_feature_split_index = index_f;
    leaf.is_leaf = false;

    // splits observations using split details
    // collects all observations already assigned from subset

    Vector<Integer> left_obs = new Vector<>();
    Vector<Integer> right_obs = new Vector<>();

    if (best_candidates_for_left_after_split.size() > 0) {
      it = best_candidates_for_left_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();
        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0))
          for (i = 0;
              i < ms.generative_support.local_empirical_measure.observations_indexes.length;
              i++)
            left_obs.addElement(
                new Integer(ms.generative_support.local_empirical_measure.observations_indexes[i]));
      }
    }

    if (best_candidates_for_right_after_split.size() > 0) {
      it = best_candidates_for_right_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();
        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0))
          for (i = 0;
              i < ms.generative_support.local_empirical_measure.observations_indexes.length;
              i++)
            right_obs.addElement(
                new Integer(ms.generative_support.local_empirical_measure.observations_indexes[i]));
      }
    }

    Vector<Integer> all_leaf_obs = new Vector<>();
    int[] alo;

    for (i = 0; i < leaf.observation_indexes_in_node.length; i++)
      all_leaf_obs.addElement(new Integer(leaf.observation_indexes_in_node[i]));

    int[][] new_split_observations = new int[2][];

    new_split_observations[0] = new int[left_obs.size()];
    new_split_observations[1] = new int[right_obs.size()];

    for (i = 0; i < left_obs.size(); i++)
      new_split_observations[0][i] = left_obs.elementAt(i).intValue();

    for (i = 0; i < right_obs.size(); i++)
      new_split_observations[1][i] = right_obs.elementAt(i).intValue();

    // remove subset_with_leaf from tol
    it = subset_with_leaf.iterator();
    while (it.hasNext()) {
      ms = (MeasuredSupportAtTupleOfNodes) it.next();
      found = tol.remove(ms);
      if (!found)
        Dataset.perror(
            "Algorithm.class :: subset of support " + ms + " should be in the set of all");
    }

    // update tol: add all elements in best_candidates_for_left_after_split &
    // best_candidates_for_right_after_split WITH >0 MEASURE, remove from tol all elements in
    // subset;

    if (best_candidates_for_left_after_split.size() > 0) {
      it = best_candidates_for_left_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();

        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0)) {
          ms.marked_for_update = true;
          ms.marked_for_update_index_tree = leaf.myTree.name;
          ms.marked_for_update_which_child = MeasuredSupportAtTupleOfNodes.LEFT_CHILD;

          tol.add(ms);
        }
      }
    }

    if (best_candidates_for_right_after_split.size() > 0) {
      it = best_candidates_for_right_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();

        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0)) {
          ms.marked_for_update = true;
          ms.marked_for_update_index_tree = leaf.myTree.name;
          ms.marked_for_update_which_child = MeasuredSupportAtTupleOfNodes.RIGHT_CHILD;

          tol.add(ms);
        }
      }
    }

    return new SplitDetails(index_f, index_s, new_split_observations);
  }

  public SplitDetails split_top_down_boosting_generative_forest_fast(
      Node leaf,
      boolean[] splittable_feature,
      FeatureTest[][] all_feature_tests,
      HashSet<MeasuredSupportAtTupleOfNodes> tol) {
    // tol contains all current tuple of leaves whose intersection support has >0 empirical measure
    // fast method which does not record where observations with missing values go (to match the
    // argument used to compute Bayes risk during testing all splits and that for the split
    // selected)

    HashSet<MeasuredSupportAtTupleOfNodes> subset_with_leaf = new HashSet<>();
    HashSet<MeasuredSupportAtTupleOfNodes> new_candidates_after_split;
    HashSet<MeasuredSupportAtTupleOfNodes> best_candidates_for_left_after_split = null;
    HashSet<MeasuredSupportAtTupleOfNodes> best_candidates_for_right_after_split = null;
    Iterator it = tol.iterator();
    MeasuredSupportAtTupleOfNodes ms;

    int number_obs_tot = 0,
        i,
        j,
        k,
        index_f = -1,
        index_s = -1,
        number_left_obs,
        number_right_obs,
        iter = 0,
        index_try;
    double cur_Bayes, p_ccstar, opt_Bayes = -1.0;
    boolean found, at_least_one;

    double[] rapid_split_stats = new double[4];

    while (it.hasNext()) {
      ms = (MeasuredSupportAtTupleOfNodes) it.next();
      if (ms.tree_nodes_support_contains_node(leaf)) {
        subset_with_leaf.add(ms);
        number_obs_tot += ms.generative_support.local_empirical_measure.observations_indexes.length;
      }
    }

    if (number_obs_tot != leaf.observation_indexes_in_node.length)
      Dataset.perror(
          "Algorithm.class :: mismatch between the total #observations ("
              + number_obs_tot
              + ") and leaf weight ("
              + leaf.observation_indexes_in_node.length
              + ")");

    if (subset_with_leaf.size() == 0)
      Dataset.perror(
          "Algorithm.class :: no subsets of MeasuredSupportAtTupleOfNodes with leaf " + leaf);

    // speed up
    Vector<Integer[]> couple_feature_index_feature_test_index = new Vector<>();

    for (i = 0; i < splittable_feature.length; i++)
      if (splittable_feature[i])
        for (j = 0; j < all_feature_tests[i].length; j++)
          couple_feature_index_feature_test_index.addElement(
              new Integer[] {new Integer(i), new Integer(j)});

    List<Integer> shuffled_indexes = new ArrayList<>();
    for (i = 0; i < couple_feature_index_feature_test_index.size(); i++) shuffled_indexes.add(i);
    Collections.shuffle(shuffled_indexes);

    int
        max_number_tries =
            (couple_feature_index_feature_test_index.size()
                    > Algorithm.MAXIMAL_NUMBER_OF_SPLIT_TESTS_TRIES_PER_BOOSTING_ITERATION)
                ? Algorithm.MAXIMAL_NUMBER_OF_SPLIT_TESTS_TRIES_PER_BOOSTING_ITERATION
                : couple_feature_index_feature_test_index.size(),
        dumi;

    // works regardless of the option chosen

    found = false;
    for (k = 0; k < max_number_tries; k++) {
      index_try = shuffled_indexes.get(shuffled_indexes.size() - 1).intValue();
      i = couple_feature_index_feature_test_index.elementAt(index_try)[0].intValue();
      j = couple_feature_index_feature_test_index.elementAt(index_try)[1].intValue();

      dumi = shuffled_indexes.remove(shuffled_indexes.size() - 1);

      iter++;

      new_candidates_after_split = new HashSet<>();
      cur_Bayes = 0.0;

      // computes all split_top_down_boosting_statistics & current Bayes entropy (HL in (7))
      number_left_obs = number_right_obs = 0;
      it = subset_with_leaf.iterator();
      at_least_one = false;
      while (it.hasNext()) {
        at_least_one = true;
        ms = (MeasuredSupportAtTupleOfNodes) it.next();

        rapid_split_stats = ms.rapid_split_statistics(all_feature_tests[i][j], i);

        number_left_obs += rapid_split_stats[0];
        number_right_obs += rapid_split_stats[1];

        p_ccstar =
            (Statistics.PRIOR
                    * ((double)
                        ms.generative_support.local_empirical_measure.observations_indexes.length)
                    / ((double) myDS.observations_from_file.size()))
                + ((1.0 - Statistics.PRIOR)
                    * ms.generative_support.support.weight_uniform_distribution);

        cur_Bayes +=
            (p_ccstar
                * Statistics.MU_BAYES_GENERATIVE_FOREST_SIMPLE(
                    rapid_split_stats[0],
                    rapid_split_stats[1],
                    rapid_split_stats[2],
                    rapid_split_stats[3],
                    myDS.observations_from_file.size(),
                    1.0));
      }

      // Bayes entropy improvement ?
      if ((at_least_one)
          && (number_left_obs >= Algorithm.MINIMUM_EMPIRICAL_CARD_AT_NEW_LEAF_FOR_BOOSTING)
          && (number_right_obs >= Algorithm.MINIMUM_EMPIRICAL_CARD_AT_NEW_LEAF_FOR_BOOSTING)
          && ((!found) || (cur_Bayes < opt_Bayes))) { //
        index_f = i;
        index_s = j;
        opt_Bayes = cur_Bayes;
        found = true;
      }
    }

    // update leaf
    if ((index_f == -1) || (index_s == -1)) return null;

    // compute best_candidates_for_left_after_split, best_candidates_for_right_after_split : SPEED
    // UP
    it = subset_with_leaf.iterator();
    best_candidates_for_left_after_split = new HashSet<>();
    best_candidates_for_right_after_split = new HashSet<>();
    while (it.hasNext()) {
      ms = (MeasuredSupportAtTupleOfNodes) it.next();
      ms.split_top_down_boosting_statistics =
          ms.split_for_boosting_computations(all_feature_tests[index_f][index_s], index_f);
      if (ms.split_top_down_boosting_statistics[0] != null)
        best_candidates_for_left_after_split.add(ms.split_top_down_boosting_statistics[0]);
      if (ms.split_top_down_boosting_statistics[1] != null)
        best_candidates_for_right_after_split.add(ms.split_top_down_boosting_statistics[1]);
    }
    // end of SPEED UP

    leaf.node_feature_test = all_feature_tests[index_f][index_s];
    leaf.node_feature_split_index = index_f;
    leaf.is_leaf = false;

    // splits observations using split details
    // collects all observations already assigned from subset

    Vector<Integer> left_obs = new Vector<>();
    Vector<Integer> right_obs = new Vector<>();

    if (best_candidates_for_left_after_split.size() > 0) {
      it = best_candidates_for_left_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();
        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0))
          for (i = 0;
              i < ms.generative_support.local_empirical_measure.observations_indexes.length;
              i++)
            left_obs.addElement(
                new Integer(ms.generative_support.local_empirical_measure.observations_indexes[i]));
      }
    }

    if (best_candidates_for_right_after_split.size() > 0) {
      it = best_candidates_for_right_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();
        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0))
          for (i = 0;
              i < ms.generative_support.local_empirical_measure.observations_indexes.length;
              i++)
            right_obs.addElement(
                new Integer(ms.generative_support.local_empirical_measure.observations_indexes[i]));
      }
    }

    Vector<Integer> all_leaf_obs = new Vector<>();
    int[] alo;

    for (i = 0; i < leaf.observation_indexes_in_node.length; i++)
      all_leaf_obs.addElement(new Integer(leaf.observation_indexes_in_node[i]));

    int[][] new_split_observations = new int[2][];

    new_split_observations[0] = new int[left_obs.size()];
    new_split_observations[1] = new int[right_obs.size()];

    for (i = 0; i < left_obs.size(); i++)
      new_split_observations[0][i] = left_obs.elementAt(i).intValue();

    for (i = 0; i < right_obs.size(); i++)
      new_split_observations[1][i] = right_obs.elementAt(i).intValue();

    // remove subset_with_leaf from tol
    it = subset_with_leaf.iterator();
    while (it.hasNext()) {
      ms = (MeasuredSupportAtTupleOfNodes) it.next();
      found = tol.remove(ms);
      if (!found)
        Dataset.perror(
            "Algorithm.class :: subset of support " + ms + " should be in the set of all");
    }

    // update tol: add all elements in best_candidates_for_left_after_split &
    // best_candidates_for_right_after_split WITH >0 MEASURE, remove from tol all elements in
    // subset;

    if (best_candidates_for_left_after_split.size() > 0) {
      it = best_candidates_for_left_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();

        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0)) {
          ms.marked_for_update = true;
          ms.marked_for_update_index_tree = leaf.myTree.name;
          ms.marked_for_update_which_child = MeasuredSupportAtTupleOfNodes.LEFT_CHILD;

          tol.add(ms);
        }
      }
    }

    if (best_candidates_for_right_after_split.size() > 0) {
      it = best_candidates_for_right_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();

        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0)) {
          ms.marked_for_update = true;
          ms.marked_for_update_index_tree = leaf.myTree.name;
          ms.marked_for_update_which_child = MeasuredSupportAtTupleOfNodes.RIGHT_CHILD;

          tol.add(ms);
        }
      }
    }

    return new SplitDetails(index_f, index_s, new_split_observations);
  }

  public SplitDetails split_top_down_boosting_ensemble_of_generative_trees_fast(
      Node leaf,
      boolean[] splittable_feature,
      FeatureTest[][] all_feature_tests,
      HashSet<MeasuredSupportAtTupleOfNodes> tol) {
    // tol contains all current tuple of leaves whose intersection support has >0 empirical measure

    HashSet<MeasuredSupportAtTupleOfNodes> subset_with_leaf = new HashSet<>();
    HashSet<MeasuredSupportAtTupleOfNodes> new_candidates_after_split;
    HashSet<MeasuredSupportAtTupleOfNodes> best_candidates_for_left_after_split = null;
    HashSet<MeasuredSupportAtTupleOfNodes> best_candidates_for_right_after_split = null;
    Iterator it = tol.iterator();
    MeasuredSupportAtTupleOfNodes ms;

    int number_obs_tot = 0,
        i,
        j,
        k,
        l,
        index_f = -1,
        number_left_obs,
        number_right_obs,
        index_s = -1,
        iter = 0,
        index_try;
    double cur_Bayes,
        p_ccstar,
        opt_Bayes = -1.0,
        p_R,
        p_R_left,
        p_R_right,
        vol0,
        vol1,
        p0,
        p1,
        u_left,
        u_right;
    boolean found, at_least_one;

    Support[] split_support;

    double[] rapid_split_stats = new double[4];

    while (it.hasNext()) {
      ms = (MeasuredSupportAtTupleOfNodes) it.next();
      if (ms.tree_nodes_support_contains_node(leaf)) {
        subset_with_leaf.add(ms);
        number_obs_tot += ms.generative_support.local_empirical_measure.observations_indexes.length;
      }
    }

    if (number_obs_tot != leaf.observation_indexes_in_node.length)
      Dataset.perror(
          "Algorithm.class :: mismatch between the total #observations ("
              + number_obs_tot
              + ") and leaf weight ("
              + leaf.observation_indexes_in_node.length
              + ")");

    if (subset_with_leaf.size() == 0)
      Dataset.perror(
          "Algorithm.class :: no subsets of MeasuredSupportAtTupleOfNodes with leaf " + leaf);

    // speed up
    Vector<Integer[]> couple_feature_index_feature_test_index = new Vector<>();

    for (i = 0; i < splittable_feature.length; i++)
      if (splittable_feature[i])
        for (j = 0; j < all_feature_tests[i].length; j++)
          couple_feature_index_feature_test_index.addElement(
              new Integer[] {new Integer(i), new Integer(j)});

    List<Integer> shuffled_indexes = new ArrayList<>();
    for (i = 0; i < couple_feature_index_feature_test_index.size(); i++) shuffled_indexes.add(i);
    Collections.shuffle(shuffled_indexes);

    int
        max_number_tries =
            (couple_feature_index_feature_test_index.size()
                    > Algorithm.MAXIMAL_NUMBER_OF_SPLIT_TESTS_TRIES_PER_BOOSTING_ITERATION)
                ? Algorithm.MAXIMAL_NUMBER_OF_SPLIT_TESTS_TRIES_PER_BOOSTING_ITERATION
                : couple_feature_index_feature_test_index.size(),
        dumi;

    // works regardless of the option chosen

    found = false;
    for (k = 0; k < max_number_tries; k++) {
      index_try = shuffled_indexes.get(shuffled_indexes.size() - 1).intValue();
      i = couple_feature_index_feature_test_index.elementAt(index_try)[0].intValue();
      j = couple_feature_index_feature_test_index.elementAt(index_try)[1].intValue();

      dumi = shuffled_indexes.remove(shuffled_indexes.size() - 1);

      iter++;

      new_candidates_after_split = new HashSet<>();
      cur_Bayes = 0.0;

      // computes all split_top_down_boosting_statistics & current Bayes entropy (HL in (7))
      number_left_obs = number_right_obs = 0;

      it = subset_with_leaf.iterator();
      at_least_one = false;
      while (it.hasNext()) {
        at_least_one = true;
        ms = (MeasuredSupportAtTupleOfNodes) it.next();

        rapid_split_stats = ms.rapid_split_statistics(all_feature_tests[i][j], i);

        number_left_obs += rapid_split_stats[0];
        number_right_obs +=
            rapid_split_stats[
                1]; // "cheating" on cardinals just to make sure supports not empty (not used in
        // ranking splits)

        l =
            Statistics.R.nextInt(
                ms.tree_nodes_support
                    .size()); // very fast approx. should take into account all trees BUT this eases
        // distributed training

        p_R =
            (ms.tree_nodes_support.elementAt(l).p_reach
                    / ms.tree_nodes_support.elementAt(l).node_support.volume)
                * ms.generative_support.support.volume;

        split_support =
            Support.SPLIT_SUPPORT(myDS, ms.generative_support.support, all_feature_tests[i][j], i);

        if (split_support[0] != null) vol0 = (double) split_support[0].volume;
        else vol0 = 0.0;

        if (split_support[1] != null) vol1 = (double) split_support[1].volume;
        else vol1 = 0.0;

        p0 = vol0 / (vol0 + vol1);
        p1 = vol1 / (vol0 + vol1);

        p_R_left = p_R * p0;
        p_R_right = p_R * p1;

        u_left = ms.generative_support.support.weight_uniform_distribution * p0;
        u_right = ms.generative_support.support.weight_uniform_distribution * p1;

        p_ccstar =
            (Statistics.PRIOR * p_R)
                + ((1.0 - Statistics.PRIOR)
                    * ms.generative_support.support.weight_uniform_distribution);

        cur_Bayes +=
            (p_ccstar
                * Statistics.MU_BAYES_ENSEMBLE_OF_GENERATIVE_TREES(
                    myDS, p_R, p_R_left, p_R_right, u_left, u_right, 1.0));
      }

      // Bayes entropy improvement ?
      if ((at_least_one)
          && (number_left_obs >= Algorithm.MINIMUM_EMPIRICAL_CARD_AT_NEW_LEAF_FOR_BOOSTING)
          && (number_right_obs >= Algorithm.MINIMUM_EMPIRICAL_CARD_AT_NEW_LEAF_FOR_BOOSTING)
          && ((!found) || (cur_Bayes < opt_Bayes))) {
        index_f = i;
        index_s = j;
        opt_Bayes = cur_Bayes;
        found = true;
      }
    }

    // update leaf
    if ((index_f == -1) || (index_s == -1)) return null;

    // compute best_candidates_for_left_after_split, best_candidates_for_right_after_split : SPEED
    // UP
    it = subset_with_leaf.iterator();
    best_candidates_for_left_after_split = new HashSet<>();
    best_candidates_for_right_after_split = new HashSet<>();
    while (it.hasNext()) {
      ms = (MeasuredSupportAtTupleOfNodes) it.next();
      ms.split_top_down_boosting_statistics =
          ms.split_for_boosting_computations(all_feature_tests[index_f][index_s], index_f);
      if (ms.split_top_down_boosting_statistics[0] != null)
        best_candidates_for_left_after_split.add(ms.split_top_down_boosting_statistics[0]);
      if (ms.split_top_down_boosting_statistics[1] != null)
        best_candidates_for_right_after_split.add(ms.split_top_down_boosting_statistics[1]);
    }
    // end of SPEED UP

    leaf.node_feature_test = all_feature_tests[index_f][index_s];
    leaf.node_feature_split_index = index_f;
    leaf.is_leaf = false;

    // splits observations using split details
    // collects all observations already assigned from subset

    Vector<Integer> left_obs = new Vector<>();
    Vector<Integer> right_obs = new Vector<>();

    if (best_candidates_for_left_after_split.size() > 0) {
      it = best_candidates_for_left_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();
        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0))
          for (i = 0;
              i < ms.generative_support.local_empirical_measure.observations_indexes.length;
              i++)
            left_obs.addElement(
                new Integer(ms.generative_support.local_empirical_measure.observations_indexes[i]));
      }
    }

    if (best_candidates_for_right_after_split.size() > 0) {
      it = best_candidates_for_right_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();
        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0))
          for (i = 0;
              i < ms.generative_support.local_empirical_measure.observations_indexes.length;
              i++)
            right_obs.addElement(
                new Integer(ms.generative_support.local_empirical_measure.observations_indexes[i]));
      }
    }

    Vector<Integer> all_leaf_obs = new Vector<>();
    int[] alo;

    for (i = 0; i < leaf.observation_indexes_in_node.length; i++)
      all_leaf_obs.addElement(new Integer(leaf.observation_indexes_in_node[i]));

    int[][] new_split_observations = new int[2][];

    new_split_observations[0] = new int[left_obs.size()];
    new_split_observations[1] = new int[right_obs.size()];

    for (i = 0; i < left_obs.size(); i++)
      new_split_observations[0][i] = left_obs.elementAt(i).intValue();

    for (i = 0; i < right_obs.size(); i++)
      new_split_observations[1][i] = right_obs.elementAt(i).intValue();

    // remove subset_with_leaf from tol
    it = subset_with_leaf.iterator();
    while (it.hasNext()) {
      ms = (MeasuredSupportAtTupleOfNodes) it.next();
      found = tol.remove(ms);
      if (!found)
        Dataset.perror(
            "Algorithm.class :: subset of support " + ms + " should be in the set of all");
    }

    // update tol: add all elements in best_candidates_for_left_after_split &
    // best_candidates_for_right_after_split WITH >0 MEASURE, remove from tol all elements in
    // subset;

    if (best_candidates_for_left_after_split.size() > 0) {
      it = best_candidates_for_left_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();

        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0)) {
          ms.marked_for_update = true;
          ms.marked_for_update_index_tree = leaf.myTree.name;
          ms.marked_for_update_which_child = MeasuredSupportAtTupleOfNodes.LEFT_CHILD;

          tol.add(ms);
        }
      }
    }

    if (best_candidates_for_right_after_split.size() > 0) {
      it = best_candidates_for_right_after_split.iterator();
      while (it.hasNext()) {
        ms = (MeasuredSupportAtTupleOfNodes) it.next();

        if ((ms.generative_support.local_empirical_measure.observations_indexes != null)
            && (ms.generative_support.local_empirical_measure.observations_indexes.length > 0)) {
          ms.marked_for_update = true;
          ms.marked_for_update_index_tree = leaf.myTree.name;
          ms.marked_for_update_which_child = MeasuredSupportAtTupleOfNodes.RIGHT_CHILD;

          tol.add(ms);
        }
      }
    }

    return new SplitDetails(index_f, index_s, new_split_observations);
  }
}
