// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class WeightedSupportAtTupleOfNodes and related classes
 * Mainly for EOGT
 *****/

class WeightedSupportAtTupleOfNodes implements Debuggable {
  // used for EOGT
  // just a support with a total weight

  public WeightedSupport generative_support;

  Vector<Node> tree_nodes_support;
  // set of nodes whose support intersection makes the support
  // (i) missing data imputation: progressively flushed out (empty iff all nodes are leaves), then
  // stored and used to impute
  // (ii) boosting: not flushed out

  WeightedSupportAtTupleOfNodes[] split_top_down_boosting_statistics;

  boolean marked_for_update;
  // FLAG used for boosting: after having updated the Hashset of supports, go through the (new)
  // MeasuredSupportAtTupleOfNodes and updates the relevant old leaf with new child = leaf
  int marked_for_update_index_tree;
  String marked_for_update_which_child;
  MeasuredSupportAtTupleOfNodes marked_for_update_parent_msatol;

  GenerativeModelBasedOnEnsembleOfTrees geot;

  WeightedSupportAtTupleOfNodes(GenerativeModelBasedOnEnsembleOfTrees g) {
    // two uses
    // (i) initializes MDS to the whole domain of unknown features: filter_out_leaves = true,
    // check_all_are_leaves = false
    // (ii) top down boosting: filter_out_leaves = false, check_all_are_leaves = true
    geot = g;

    generative_support = new WeightedSupport(g); // generative_support = null;

    int i;
    split_top_down_boosting_statistics = null;
    marked_for_update = false;
    marked_for_update_parent_msatol = null;
    marked_for_update_which_child = null;
    marked_for_update_index_tree = -1;

    tree_nodes_support = new Vector<>();
    for (i = 0; i < geot.trees.size(); i++) {
      tree_nodes_support.addElement(geot.trees.elementAt(i).root);
    }
  }

  public static void CHECK_BOOSTING_CONSISTENCY(
      GenerativeModelBasedOnEnsembleOfTrees g, HashSet<WeightedSupportAtTupleOfNodes> tol) {
    if ((tol == null) || (tol.size() == 0)) return;

    Iterator it = tol.iterator();
    WeightedSupportAtTupleOfNodes ms;

    while (it.hasNext()) {
      ms = (WeightedSupportAtTupleOfNodes) it.next();
      if (ms.tree_nodes_support.size() != g.trees.size())
        Dataset.perror(
            "WeightedSupportAtTupleOfNodes.class :: inconsistent WeightedSupportAtTupleOfNodes");
    }
  }

  public static WeightedSupportAtTupleOfNodes copyOf(WeightedSupportAtTupleOfNodes mds) {
    WeightedSupportAtTupleOfNodes ret = new WeightedSupportAtTupleOfNodes(mds.geot);

    ret.generative_support = WeightedSupport.copyOf(mds.generative_support);
    ret.tree_nodes_support = new Vector<>();
    int i;
    for (i = 0; i < mds.tree_nodes_support.size(); i++)
      ret.tree_nodes_support.addElement(mds.tree_nodes_support.elementAt(i));

    return ret;
  }

  public void unmark_for_update(Node new_left_node, Node new_right_node) {
    if ((!marked_for_update)
        || (marked_for_update_index_tree == -1)
        || (marked_for_update_which_child.equals(MeasuredSupportAtTupleOfNodes.NO_CHILD)))
      Dataset.perror(
          "WeightedSupportAtTupleOfNodes.class :: unmarking an unmarked"
              + " WeightedSupportAtTupleOfNodes");

    if (marked_for_update_which_child.equals(MeasuredSupportAtTupleOfNodes.LEFT_CHILD))
      tree_nodes_support.setElementAt(new_left_node, marked_for_update_index_tree);
    else if (marked_for_update_which_child.equals(MeasuredSupportAtTupleOfNodes.RIGHT_CHILD))
      tree_nodes_support.setElementAt(new_right_node, marked_for_update_index_tree);
    else
      Dataset.perror(
          "WeightedSupportAtTupleOfNodes.class :: no such token as "
              + marked_for_update_which_child);

    marked_for_update_parent_msatol = null;
    marked_for_update = false;
    marked_for_update_index_tree = -1;
    marked_for_update_which_child = MeasuredSupportAtTupleOfNodes.NO_CHILD;
  }

  public boolean tree_nodes_support_contains_node(Node n) {
    if (tree_nodes_support.elementAt(n.myTree.name).equals(n)) return true;
    return false;
  }

  public void check_all_tree_nodes_support_are_leaves() {
    int i;
    for (i = 0; i < tree_nodes_support.size(); i++)
      if (!tree_nodes_support.elementAt(i).is_leaf)
        Dataset.perror(
            "MeasuredSupportAtTupleOfNodes.class :: not all leaves ("
                + tree_nodes_support.elementAt(i)
                + ")");
  }

  public boolean done_for_missing_data_imputation() {
    return (tree_nodes_support.size() == 0);
  }

  public void prepare_for_missing_data_imputation() {
    int i;
    for (i = tree_nodes_support.size() - 1; i >= 0; i--)
      if (tree_nodes_support.elementAt(i).is_leaf) tree_nodes_support.removeElementAt(i);
  }

  public void squash_for_missing_data_imputation(Observation o) {
    // reduces to null all features that are NOT unknown in o
    int i;
    for (i = 0; i < generative_support.support.dim(); i++)
      if (!Observation.FEATURE_IS_UNKNOWN(o, i))
        generative_support.support.setNullFeatureAt(i, geot.myDS);
  }

  public String toString() {
    String ret = "";
    ret = generative_support + "{{" + tree_nodes_support.size() + ":";
    int i;
    for (i = 0; i < tree_nodes_support.size(); i++) {
      ret += tree_nodes_support.elementAt(i).name;
      if (i < tree_nodes_support.size() - 1) ret += ",";
    }
    ret += "}}";
    return ret;
  }

  public double p_R_independence() {
    double p_R = 1.0;
    int l;
    for (l = 0; l < tree_nodes_support.size(); l++) p_R *= tree_nodes_support.elementAt(l).p_reach;
    return p_R;
  }

  public WeightedSupportAtTupleOfNodes[] split_for_boosting_computations(
      FeatureTest ft, int feature_split_index) {
    // applies ft on generative_support.support
    // returns the left and right MeasuredSupportAtTupleOfNodes WITH NODE UNCHANGED (has to be
    // changed by left / right child afterwards WHEN FEATURE CHOSEN IN BOOSTING)
    // completes otherwise everything: new supports, examples at the supports
    // IF one MeasuredSupportAtTupleOfNodes has no example reaching it, replaces it by null
    // (simplifies boosting related computations)

    Feature feature_in_measured_support = generative_support.support.feature(feature_split_index);

    WeightedSupportAtTupleOfNodes[] ret = new WeightedSupportAtTupleOfNodes[2];

    LocalEmpiricalMeasure[] split_meas;
    Feature[] split_feat;

    FeatureTest f = FeatureTest.copyOf(ft, feature_in_measured_support);

    String tvb = f.check_trivial_branching(geot.myDS, feature_in_measured_support, true);

    if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)) {
      ret[0] = WeightedSupportAtTupleOfNodes.copyOf(this); // left
      ret[0].check_all_tree_nodes_support_are_leaves();

      ret[1] = null; // right
    } else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)) {
      ret[0] = null; // left

      ret[1] = WeightedSupportAtTupleOfNodes.copyOf(this); // right
      ret[1].check_all_tree_nodes_support_are_leaves();
    } else {
      ret[0] = WeightedSupportAtTupleOfNodes.copyOf(this); // left
      ret[0].check_all_tree_nodes_support_are_leaves();

      ret[1] = WeightedSupportAtTupleOfNodes.copyOf(this); // right
      ret[1].check_all_tree_nodes_support_are_leaves();

      split_feat = f.split_feature(geot.myDS, feature_in_measured_support, false, false);

      ret[0].generative_support.support.setFeatureAt(split_feat[0], feature_split_index);
      ret[1].generative_support.support.setFeatureAt(split_feat[1], feature_split_index);
    }

    return ret;
  }

  public WeightedSupportAtTupleOfNodes[]
      split_for_missing_data_imputation_ensemble_of_generative_trees(Observation reference) {
    // same as split_for_missing_data_imputation_generative_forest but uses
    // WeightedSupportAtTupleOfNodes for Generative Trees (faster)

    if (done_for_missing_data_imputation())
      Dataset.perror("WeightedSupportAtTupleOfNodes.class :: cannot try splitting: MDS finished");

    int element_chosen = 0;

    Node tree_node = tree_nodes_support.elementAt(element_chosen);

    int index_feature_split = tree_node.node_feature_split_index;

    Feature feature_in_tree_node = tree_node.node_support.feature(index_feature_split);
    Feature feature_in_measured_support = generative_support.support.feature(index_feature_split);

    WeightedSupportAtTupleOfNodes[] ret;

    Feature[] split_feat;
    boolean goes_left, zero_measure, no_left = false, no_right = false;

    double p_hat_left, p_hat_right, fact_left, fact_right;
    // see paper (10)

    FeatureTest f = FeatureTest.copyOf(tree_node.node_feature_test, feature_in_tree_node);

    if (!Observation.FEATURE_IS_UNKNOWN(reference, index_feature_split)) {
      goes_left =
          tree_node.node_feature_test.observation_goes_left(
              reference, geot.myDS, feature_in_tree_node, true);
      if (goes_left) {
        tree_nodes_support.setElementAt(tree_node.left_child, element_chosen);
      } else {
        tree_nodes_support.setElementAt(tree_node.right_child, element_chosen);
      }
      if (tree_nodes_support.elementAt(element_chosen).is_leaf)
        tree_nodes_support.removeElementAt(element_chosen);

      ret = null;
    } else {
      String tvb = f.check_trivial_branching(geot.myDS, feature_in_measured_support, false);

      if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)) {
        tree_nodes_support.setElementAt(tree_node.left_child, element_chosen);

        if (tree_nodes_support.elementAt(element_chosen).is_leaf)
          tree_nodes_support.removeElementAt(element_chosen);

        ret = null;
      } else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)) {
        tree_nodes_support.setElementAt(tree_node.right_child, element_chosen);

        if (tree_nodes_support.elementAt(element_chosen).is_leaf)
          tree_nodes_support.removeElementAt(element_chosen);

        ret = null;
      } else {
        split_feat = f.split_feature(geot.myDS, feature_in_measured_support, false, false);
        if ((split_feat[0].length() == 0.0) && (split_feat[1].length() == 0.0))
          Dataset.perror(
              "WeightedSupportAtTupleOfNodes.class :: splitting a feature gives two features w/ 0"
                  + " length");

        double ntn, tnn, vtildet, vtildef;

        ret = new WeightedSupportAtTupleOfNodes[2];
        if (split_feat[0].length() == 0.0) {
          ret[0] = null;
          vtildef = 0.0;
        } else {
          ret[0] = WeightedSupportAtTupleOfNodes.copyOf(this);
          if (tree_node.left_child.is_leaf)
            ret[0].tree_nodes_support.removeElementAt(element_chosen);
          else ret[0].tree_nodes_support.setElementAt(tree_node.left_child, element_chosen);
          ret[0].generative_support.support.setFeatureAt(split_feat[0], index_feature_split);
          vtildef = ret[0].generative_support.support.volume;
        }

        if (split_feat[1].length() == 0.0) {
          ret[1] = null;
          vtildet = 0.0;
        } else {
          ret[1] = WeightedSupportAtTupleOfNodes.copyOf(this);
          if (tree_node.right_child.is_leaf)
            ret[1].tree_nodes_support.removeElementAt(element_chosen);
          else ret[1].tree_nodes_support.setElementAt(tree_node.right_child, element_chosen);
          ret[1].generative_support.support.setFeatureAt(split_feat[1], index_feature_split);
          vtildet = ret[1].generative_support.support.volume;
        }

        ntn = tree_node.left_child.node_support.volume * vtildet * tree_node.right_child.p_reach;
        tnn = vtildef * tree_node.right_child.node_support.volume * tree_node.left_child.p_reach;

        if (ret[0] != null)
          ret[0].generative_support.total_weight =
              generative_support.total_weight * tnn / (ntn + tnn);

        if (ret[1] != null)
          ret[1].generative_support.total_weight =
              generative_support.total_weight * ntn / (ntn + tnn);
      }
    }
    return ret;
  }
}

class WeightedSupport implements Debuggable {
  // used to generate data with GenerativeModelBasedOnEnsembleOfTrees models
  // for EOGT

  public Support support;
  double total_weight;

  GenerativeModelBasedOnEnsembleOfTrees myGET;

  WeightedSupport(GenerativeModelBasedOnEnsembleOfTrees geot, Support s) {
    myGET = geot;
    support = s;

    total_weight = -1.0;
  }

  WeightedSupport(GenerativeModelBasedOnEnsembleOfTrees geot) {
    this(geot, geot.myDS.domain_support());
  }

  public static WeightedSupport copyOf(WeightedSupport gs) {
    WeightedSupport g = new WeightedSupport(gs.myGET, Support.copyOf(gs.support));
    g.total_weight = gs.total_weight;
    return g;
  }

  public String toString() {
    String ret = "";
    ret += support;
    ret += "{" + total_weight + "}";
    return ret;
  }
}
