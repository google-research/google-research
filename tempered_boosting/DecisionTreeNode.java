import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class DecisionTreeNode
 *****/

class DecisionTreeNode implements Debuggable {
  public static String LEFT_CHILD = "LEFT_CHILD", RIGHT_CHILD = "RIGHT_CHILD";
  int name, depth, split_CV;
  // index of the node in the tree, the root is labeled 0; depth = depth in the tree (0 = root);
  // split_CV = # fold for training

  DecisionTreeNode left_child, right_child;
  boolean is_leaf;
  // if true, left_child = right_child = null;

  double node_prediction_from_boosting_weights;
  // real valued prediction, to be used only for leaves

  double node_prediction_from_cardinals;
  // uses pos and neg only

  // OR to build the hyperbolic representation

  int[] train_fold_indexes_in_node;
  // index of examples in the training fold that reach the node
  int pos, neg;
  // #pos and neg examples in the training fold reaching the node (CARDINALS)
  double wpos_tempered, wneg_tempered;
  // sum of weights of pos and neg examples in the training fold reaching the node, tempered
  double wpos_codensity, wneg_codensity;
  // sum of weights of pos and neg examples in the training fold reaching the node, co-density

  boolean computed;
  // true iff the node has been fully computed as internal (left and right children) or leaf

  // Feature information at the node level
  DecisionTree myTree;
  int feature_node_index;
  // handle on the feature in
  // myTree.myBoost.myDomain.myDS.index_observation_features_to_index_features
  int feature_node_test_index;
  // index of the test in the feature tests vector, as recorded in handle in
  // myTree.myBoost.myDomain.myDS.index_observation_features_to_index_features
  // =tie for a test on continuous values ( <= is left, > is right)
  // =tie for a test on nominal values ( in the set is left, otherwise is right)

  Vector<Vector<Integer>> continuous_features_indexes_for_split;

  // records two int for each continuous features (else null), start and end indexes in f.tests that
  // are relevant to node

  public boolean equals(Object o) {
    if (o == this) return true;
    if (!(o instanceof DecisionTreeNode)) return false;
    DecisionTreeNode ft = (DecisionTreeNode) o;

    if ((ft.name != name)
        || (ft.depth != depth)
        || (ft.is_leaf != is_leaf)
        || (ft.myTree.name != myTree.name)
        || (ft.feature_node_index != feature_node_index)
        || (ft.feature_node_test_index != feature_node_test_index)) return false;

    return true;
  }

  DecisionTreeNode() {
    name = -1;
    computed = false;

    left_child = right_child = null;
    train_fold_indexes_in_node = null;
    is_leaf = true;

    node_prediction_from_boosting_weights = node_prediction_from_cardinals = 0.0;

    feature_node_index = -1;
    feature_node_test_index = -1;

    continuous_features_indexes_for_split = null;
  }

  DecisionTreeNode(
      DecisionTree t,
      int v,
      int d,
      int split,
      Vector indexes,
      int p,
      int n,
      double wp_tempered,
      double wn_tempered,
      double wp_codensity,
      double wn_codensity) {
    this();
    myTree = t;
    name = v;
    depth = d;
    split_CV = split;

    int i;
    if (indexes != null) {
      train_fold_indexes_in_node = new int[indexes.size()];
      for (i = 0; i < indexes.size(); i++) {
        train_fold_indexes_in_node[i] = ((Integer) indexes.elementAt(i)).intValue();
      }
    }
    pos = p;
    neg = n;

    wpos_tempered = wp_tempered;
    wneg_tempered = wn_tempered;

    wpos_codensity = wp_codensity;
    wneg_codensity = wn_codensity;
  }

  public void init_continuous_features_indexes_for_split() {
    int i;
    Vector<Integer> dumv;
    continuous_features_indexes_for_split = new Vector<>();
    for (i = 0; i < myTree.myBoost.myDomain.myDS.features.size(); i++) {
      if ((i != myTree.myBoost.myDomain.myDS.index_class)
          && (Feature.IS_CONTINUOUS(
              ((Feature) myTree.myBoost.myDomain.myDS.features.elementAt(i)).type))) {
        dumv = new Vector<Integer>();
        dumv.addElement(new Integer(0));
        dumv.addElement(
            new Integer(
                ((Feature) myTree.myBoost.myDomain.myDS.features.elementAt(i)).tests.size()));
        continuous_features_indexes_for_split.addElement(dumv);
      } else continuous_features_indexes_for_split.addElement(null);
    }
  }

  public void continuous_features_indexes_for_split_copy_from(DecisionTreeNode n) {
    continuous_features_indexes_for_split = new Vector<>();
    int i;
    Vector<Integer> dumv;
    for (i = 0; i < n.continuous_features_indexes_for_split.size(); i++) {
      if (n.continuous_features_indexes_for_split.elementAt(i) == null)
        continuous_features_indexes_for_split.addElement(null);
      else {
        dumv = new Vector<Integer>();
        dumv.addElement(
            new Integer(
                n.continuous_features_indexes_for_split.elementAt(i).elementAt(0).intValue()));
        dumv.addElement(
            new Integer(
                n.continuous_features_indexes_for_split.elementAt(i).elementAt(1).intValue()));
        continuous_features_indexes_for_split.addElement(dumv);
      }
    }
  }

  public void continuous_features_indexes_for_split_update_child(
      int f_index, int f_test_index, String which_child) {
    // updates the list of relevant tests for feature f_index, given that the node is a new child
    // continuous_features_indexes_for_split *supposed to be copied from parent*

    if ((f_index != myTree.myBoost.myDomain.myDS.index_class)
        && (Feature.IS_CONTINUOUS(
            ((Feature) myTree.myBoost.myDomain.myDS.features.elementAt(f_index)).type))) {
      if (continuous_features_indexes_for_split.elementAt(f_index) == null)
        Dataset.perror("DecisionTreeNode.class :: no feature index stored for feature #" + f_index);

      if ((f_test_index
              < continuous_features_indexes_for_split.elementAt(f_index).elementAt(0).intValue())
          || (f_test_index
              > continuous_features_indexes_for_split.elementAt(f_index).elementAt(1).intValue()))
        Dataset.perror(
            "DecisionTreeNode.class :: test index for split #"
                + f_test_index
                + " not in interval ["
                + continuous_features_indexes_for_split.elementAt(f_index).elementAt(0)
                + ","
                + continuous_features_indexes_for_split.elementAt(f_index).elementAt(1)
                + "]");

      if ((continuous_features_indexes_for_split.elementAt(f_index).elementAt(0)
              == continuous_features_indexes_for_split.elementAt(f_index).elementAt(1))
          || ((f_test_index
                  == continuous_features_indexes_for_split
                      .elementAt(f_index)
                      .elementAt(0)
                      .intValue())
              && (which_child.equals(LEFT_CHILD)))
          || ((f_test_index
                  == continuous_features_indexes_for_split
                      .elementAt(f_index)
                      .elementAt(1)
                      .intValue())
              && (which_child.equals(RIGHT_CHILD)))) {
        if ((continuous_features_indexes_for_split.elementAt(f_index).elementAt(0)
                == continuous_features_indexes_for_split.elementAt(f_index).elementAt(1))
            && (f_test_index
                != continuous_features_indexes_for_split.elementAt(f_index).elementAt(0)))
          Dataset.perror(
              "DecisionTreeNode.class :: inconsistency for feature #"
                  + f_index
                  + ": the test "
                  + f_test_index
                  + " != singleton "
                  + continuous_features_indexes_for_split.elementAt(f_index).elementAt(0));
        continuous_features_indexes_for_split.setElementAt(
            null, f_index); // no more tests to be tried for this feature
      } else {
        if (which_child.equals(LEFT_CHILD))
          continuous_features_indexes_for_split
              .elementAt(f_index)
              .setElementAt(new Integer(f_test_index - 1), 1);
        else if (which_child.equals(RIGHT_CHILD))
          continuous_features_indexes_for_split
              .elementAt(f_index)
              .setElementAt(new Integer(f_test_index + 1), 0);
        else Dataset.perror("DecisionTreeNode.class :: no tag " + which_child);
      }
    }
  }

  public void checkForOutput() {
    if (!is_leaf) Dataset.perror("DecisionTreeNode.class :: node " + this + " is not a leaf");

    if (!computed)
      Dataset.perror("DecisionTreeNode.class :: node " + this + " has link value not computed");

    if ((wpos_tempered == 0.0) && (wneg_tempered == 0.0))
      Dataset.perror("DecisionTreeNode.class :: node " + this + " has zero local tempered weights");
  }

  public void checkForOutput_MonotonicTreeGraph() {
    if (!computed)
      Dataset.perror("DecisionTreeNode.class :: node " + this + " has link value not computed");

    if ((wpos_tempered == 0.0) && (wneg_tempered == 0.0))
      Dataset.perror("DecisionTreeNode.class :: node " + this + " has zero local tempered weights");
  }

  public void compute_prediction(double tt) {
    compute_prediction(tt, true);
    compute_prediction(tt, false);
  }

  public void compute_prediction(double tt, boolean weights_or_cardinals) {
    // weights_or_cardinals = true <=> weights, else use cardinals

    double wpos_used = wpos_codensity, wneg_used = wneg_codensity, value_node_prediction = 0.0;

    if (myTree.myBoost.equals(Boost.KEY_NAME_TEMPERED_LOSS)) {
      wpos_used = wpos_codensity;
      wneg_used = wneg_codensity;
    } else {
      wpos_used = wpos_tempered;
      wneg_used = wneg_tempered;
    }

    if (!weights_or_cardinals) {
      wpos_used = (double) pos;
      wneg_used = (double) neg;
    }

    if ((wpos_used == 0.0) && (wneg_used == 0.0))
      Dataset.perror("DecisionTreeNode.class :: no weight at node " + this);

    if ((wpos_used < 0.0) || (wneg_used < 0.0))
      Dataset.perror("DecisionTreeNode.class :: wrong loss used at node " + this);

    if (wpos_used == wneg_used) value_node_prediction = 0.0;
    else {
      if ((wpos_used == 0.0) || (wneg_used == 0.0)) {
        if (wpos_used > 0.0) value_node_prediction = Boost.MAX_PRED_VALUE;
        else value_node_prediction = -Boost.MAX_PRED_VALUE;

        TemperedBoostException.ADD(TemperedBoostException.NUMERICAL_ISSUES_ABSENT_CLASS);
      } else {
        // System.out.println("PRED wpos_used = " + wpos_used + ", wneg_used = " + wneg_used);

        if (myTree.myBoost.name.equals(Boost.KEY_NAME_TEMPERED_LOSS))
          value_node_prediction =
              Statistics.CANONICAL_LINK_TEMPERED_BAYES_RISK(
                  tt, (wpos_used / (wpos_used + wneg_used)));
        else if (myTree.myBoost.name.equals(Boost.KEY_NAME_LOG_LOSS))
          value_node_prediction =
              Statistics.CANONICAL_LINK_LOG_LOSS_BAYES_RISK(wpos_used / (wpos_used + wneg_used));

        if ((Double.isNaN(value_node_prediction)) || (Double.isInfinite(value_node_prediction))) {
          if (wpos_used > wneg_used) value_node_prediction = Boost.MAX_PRED_VALUE;
          else value_node_prediction = -Boost.MAX_PRED_VALUE;

          TemperedBoostException.ADD(TemperedBoostException.NUMERICAL_ISSUES_INFINITE_LEAF_LABEL);
        }
      }
    }

    if (weights_or_cardinals) node_prediction_from_boosting_weights = value_node_prediction;
    else node_prediction_from_cardinals = value_node_prediction;

    // To reduce the risk of numerical errors | REMOVE if safe

    /*if (node_prediction > 0.0)
        node_prediction = 1.0;
    else if (node_prediction < 0.0)
    node_prediction = -1.0;*/

    computed = true;
  }

  public Vector bestSplit(double tt) {
    int i,
        j,
        k,
        l,
        start_index,
        end_index,
        dumi,
        pos_left,
        pos_right,
        neg_left,
        neg_right,
        index_try;
    Vector<Integer[]> couple_feature_index_feature_test_index = new Vector<>();
    double delta = 0.0,
        deltabest = -1.0,
        wpos_left_tempered,
        wpos_right_tempered,
        wneg_left_tempered,
        wneg_right_tempered,
        wpos_left_codensity,
        wpos_right_codensity,
        wneg_left_codensity,
        wneg_right_codensity;
    Vector left, right, v_inv = null, v_ret = null;
    Feature f;
    Example e;
    boolean found = false;

    for (i = 0;
        i < myTree.myBoost.myDomain.myDS.index_observation_features_to_index_features.length;
        i++)
      if ((Feature.HAS_MODALITIES(
              ((Feature) myTree.myBoost.myDomain.myDS.features.elementAt(i)).type))
          || (continuous_features_indexes_for_split.elementAt(i) != null)) {
        f =
            (Feature)
                myTree.myBoost.myDomain.myDS.features.elementAt(
                    myTree.myBoost.myDomain.myDS.index_observation_features_to_index_features[i]);

        if (Feature.HAS_MODALITIES(
            ((Feature) myTree.myBoost.myDomain.myDS.features.elementAt(i)).type)) {
          start_index = 0;
          end_index = f.tests.size();
        } else {
          start_index = continuous_features_indexes_for_split.elementAt(i).elementAt(0).intValue();
          end_index = continuous_features_indexes_for_split.elementAt(i).elementAt(1).intValue();
        }

        for (j = start_index; j < end_index; j++)
          couple_feature_index_feature_test_index.addElement(
              new Integer[] {new Integer(i), new Integer(j)});
      }

    List<Integer> shuffled_indexes = new ArrayList<>();
    for (i = 0; i < couple_feature_index_feature_test_index.size(); i++) shuffled_indexes.add(i);
    Collections.shuffle(shuffled_indexes);

    int max_number_tries =
        (couple_feature_index_feature_test_index.size() > Boost.MAX_SPLIT_TEST)
            ? Boost.MAX_SPLIT_TEST
            : couple_feature_index_feature_test_index.size();

    for (l = 0; l < max_number_tries; l++) {
      index_try = shuffled_indexes.get(shuffled_indexes.size() - 1).intValue();
      i = couple_feature_index_feature_test_index.elementAt(index_try)[0].intValue();
      j = couple_feature_index_feature_test_index.elementAt(index_try)[1].intValue();

      dumi = shuffled_indexes.remove(shuffled_indexes.size() - 1);

      left = new Vector();
      right = new Vector();

      pos_left = pos_right = neg_left = neg_right = 0;
      wpos_left_tempered = wpos_right_tempered = wneg_left_tempered = wneg_right_tempered = 0.0;
      wpos_left_codensity = wpos_right_codensity = wneg_left_codensity = wneg_right_codensity = 0.0;

      f =
          (Feature)
              myTree.myBoost.myDomain.myDS.features.elementAt(
                  myTree.myBoost.myDomain.myDS.index_observation_features_to_index_features[i]);

      for (k = 0; k < train_fold_indexes_in_node.length; k++) {
        e = myTree.myBoost.myDomain.myDS.train_example(split_CV, train_fold_indexes_in_node[k]);
        if (f.example_goes_left(e, i, j)) {
          left.addElement(new Integer(train_fold_indexes_in_node[k]));
          if (e.is_positive_noisy()) {
            wpos_left_tempered += e.current_boosting_weight;
            wpos_left_codensity += Math.pow(e.current_boosting_weight, 2.0 - tt);
            pos_left++;
          } else {
            wneg_left_tempered += e.current_boosting_weight;
            wneg_left_codensity += Math.pow(e.current_boosting_weight, 2.0 - tt);
            neg_left++;
          }
        } else {
          right.addElement(new Integer(train_fold_indexes_in_node[k]));
          if (e.is_positive_noisy()) {
            wpos_right_tempered += e.current_boosting_weight;
            wpos_right_codensity += Math.pow(e.current_boosting_weight, 2.0 - tt);
            pos_right++;
          } else {
            wneg_right_tempered += e.current_boosting_weight;
            wneg_right_codensity += Math.pow(e.current_boosting_weight, 2.0 - tt);
            neg_right++;
          }
        }
      }

      delta =
          Statistics.DELTA_BAYES_RISK_SPLIT(
              myTree.myBoost.name,
              tt,
              wpos_tempered,
              wneg_tempered,
              wpos_left_tempered,
              wneg_left_tempered,
              wpos_right_tempered,
              wneg_right_tempered);

      if (delta < 0.0)
        Dataset.perror(
            " delta "
                + delta
                + " < 0 -- tempered_t = "
                + tt
                + ", wpos_tempered = "
                + wpos_tempered
                + ", wneg_tempered = "
                + wneg_tempered
                + ", wpos_left_tempered = "
                + wpos_left_tempered
                + ", wneg_left_tempered = "
                + wneg_left_tempered
                + ", wpos_right_tempered = "
                + wpos_right_tempered
                + ", wneg_right_tempered = "
                + wneg_right_tempered);

      if ((wpos_left_tempered > 0.0)
          && (wneg_left_tempered > 0.0)
          && (wpos_right_tempered > 0.0)
          && (wneg_right_tempered > 0.0)) {

        if (!myTree.myBoost.name.equals(Boost.KEY_NAME_TEMPERED_LOSS))
          wpos_left_codensity =
              wneg_left_codensity = wpos_right_codensity = wneg_right_codensity = 1.0;

        v_inv = new Vector();

        v_inv.addElement(new Double(delta)); // 0
        v_inv.addElement(this); // 1
        v_inv.addElement(new Integer(i)); // 2
        v_inv.addElement(new Integer(j)); // 3
        v_inv.addElement(left); // 4
        v_inv.addElement(right); // 5
        v_inv.addElement(new Integer(pos_left)); // 6
        v_inv.addElement(new Integer(neg_left)); // 7
        v_inv.addElement(new Integer(pos_right)); // 8
        v_inv.addElement(new Integer(neg_right)); // 9
        v_inv.addElement(new Double(wpos_left_tempered)); // 10
        v_inv.addElement(new Double(wneg_left_tempered)); // 11
        v_inv.addElement(new Double(wpos_right_tempered)); // 12
        v_inv.addElement(new Double(wneg_right_tempered)); // 13
        v_inv.addElement(new Double(wpos_left_codensity)); // 14
        v_inv.addElement(new Double(wneg_left_codensity)); // 15
        v_inv.addElement(new Double(wpos_right_codensity)); // 16
        v_inv.addElement(new Double(wneg_right_codensity)); // 17

        if ((Algorithm.SPLITTABLE(myTree, this, v_inv)) && ((!found) || (delta > deltabest))) {
          deltabest = delta;
          v_ret = v_inv;
          found = true;
        }
      }

      left = right = null;
    }
    return v_ret;
  }

  public String display(HashSet<Integer> indexes) {
    String v = "", t, classification = "(" + DF.format(node_prediction_from_boosting_weights) + ")";
    int i;
    HashSet<Integer> dum;
    boolean bdum;

    t = "\u2501";

    for (i = 0; i < depth; i++) {
      if ((i == depth - 1) && (indexes.contains(new Integer(i)))) v += "\u2523" + t;
      else if (i == depth - 1) v += "\u2517" + t;
      else if (indexes.contains(new Integer(i))) v += "\u2503 ";
      else v += "  ";
    }

    v += toString();

    if (!is_leaf) {
      dum = new HashSet<Integer>(indexes);
      bdum = dum.add(new Integer(depth));

      if (left_child != null) v += left_child.display(dum);
      else v += "null";

      if (right_child != null) v += right_child.display(indexes);
      else v += "null";
    }

    return v;
  }

  public String BooleanTestAtEdge(boolean leftChild) {
    String ret = "";

    if (!leftChild) ret += "not ";

    ret +=
        ((Feature)
                myTree.myBoost.myDomain.myDS.features.elementAt(
                    myTree
                        .myBoost
                        .myDomain
                        .myDS
                        .index_observation_features_to_index_features[feature_node_index]))
            .display_test(feature_node_test_index);
    return ret;
  }

  public String toString() {
    String v = "",
        classification =
            "(B:"
                + DF.format(node_prediction_from_boosting_weights)
                + ", C:"
                + DF.format(node_prediction_from_cardinals)
                + ")";
    int leftn, rightn;

    if (name != 0) v += "[#" + name + "]";
    else v += "[#0:root]";

    if (is_leaf) v += " leaf " + observations_string() + " " + classification;
    else {
      if (left_child != null) leftn = left_child.name;
      else leftn = -1;

      if (right_child != null) rightn = right_child.name;
      else rightn = -1;

      v +=
          " internal ("
              + ((Feature)
                      myTree.myBoost.myDomain.myDS.features.elementAt(
                          myTree
                              .myBoost
                              .myDomain
                              .myDS
                              .index_observation_features_to_index_features[feature_node_index]))
                  .display_test(feature_node_test_index)
              + " ? #"
              + left_child.name
              + " : #"
              + right_child.name
              + ") -- classification (indicative): "
              + classification;

      if (DecisionTree.DISPLAY_INTERNAL_NODES_CLASSIFICATION) v += classification;
    }

    v += "\n";

    return v;
  }

  public String observations_string() {
    return "{"
        + pos
        + ":"
        + neg
        + "}{"
        + DF6.format(wpos_tempered)
        + ":"
        + DF6.format(wneg_tempered)
        + "}";
  }
}
