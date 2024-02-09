import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class DecisionTree
 *****/

class DecisionTree implements Debuggable {
  public static boolean DISPLAY_INTERNAL_NODES_CLASSIFICATION = false;

  int name, depth;

  DecisionTreeNode root;
  // root of the tree

  Vector leaves;
  // list of leaves of the tree (potential growth here)

  Boost myBoost;

  int max_size;

  int split_CV;

  int number_nodes;
  // includes leaves

  double tree_p_t, tree_psi_t, tree_p_t_star;

  DecisionTree(int nn, Boost bb, int maxs, int split) {
    name = nn;
    root = null;
    myBoost = bb;

    max_size = maxs;

    split_CV = split;

    tree_p_t = tree_psi_t = tree_p_t_star = -1.0;
  }

  public String toString() {
    int i;
    String v = "(name = #" + name + " | depth = " + depth + " | #nodes = " + number_nodes + ")\n";
    DecisionTreeNode dumn;

    v += root.display(new HashSet<Integer>());

    v += "Leaves:";

    Iterator it = leaves.iterator();
    while (it.hasNext()) {
      v += " ";
      dumn = (DecisionTreeNode) it.next();
      v += "#" + dumn.name + dumn.observations_string();
    }
    v += ".\n";

    return v;
  }

  public static void INSERT_WEIGHT_ORDER(Vector<DecisionTreeNode> all_nodes, DecisionTreeNode nn) {
    int k;

    if (all_nodes.size() == 0) all_nodes.addElement(nn);
    else {
      k = 0;
      while ((k < all_nodes.size())
          && (nn.train_fold_indexes_in_node.length
              < all_nodes.elementAt(k).train_fold_indexes_in_node.length)) k++;
      all_nodes.insertElementAt(nn, k);
    }
  }

  public void grow_heavy_first() {
    // grows the heaviest grow-able leaf
    // heavy = wrt the total # examples reaching the node

    Vector vin, vvv;
    boolean stop = false;
    DecisionTreeNode nn, leftnn, rightnn;

    int i, j, k, l, ibest, pos_left, neg_left, pos_right, neg_right, nsplits = 0;
    double wpos_left_tempered,
        wneg_left_tempered,
        wpos_right_tempered,
        wneg_right_tempered,
        t_leaf = myBoost.tempered_t,
        w_leaf_codensity;
    double wpos_left_codensity, wneg_left_codensity, wpos_right_codensity, wneg_right_codensity;

    Vector<DecisionTreeNode> try_leaves = null; // leaves that will be tried to grow
    Vector candidate_split, dumv;

    try_leaves = new Vector<>();
    for (j = 0; j < leaves.size(); j++)
      DecisionTree.INSERT_WEIGHT_ORDER(try_leaves, (DecisionTreeNode) leaves.elementAt(j));

    do {
      do {
        nn = ((DecisionTreeNode) try_leaves.elementAt(0));
        candidate_split = nn.bestSplit(t_leaf);

        if (candidate_split == null) try_leaves.removeElementAt(0);
      } while ((try_leaves.size() > 0) && (candidate_split == null));

      if (candidate_split == null) stop = true;
      else {
        vin = candidate_split;

        nn.is_leaf = false;
        try_leaves.removeElementAt(0);

        // nn.node_prediction_from_boosting_weights = nn.node_prediction_from_cardinals = 0.0;
        nn.feature_node_index = ((Integer) vin.elementAt(2)).intValue();
        nn.feature_node_test_index = ((Integer) vin.elementAt(3)).intValue();

        pos_left = ((Integer) vin.elementAt(6)).intValue();
        neg_left = ((Integer) vin.elementAt(7)).intValue();
        pos_right = ((Integer) vin.elementAt(8)).intValue();
        neg_right = ((Integer) vin.elementAt(9)).intValue();

        wpos_left_tempered = ((Double) vin.elementAt(10)).doubleValue();
        wneg_left_tempered = ((Double) vin.elementAt(11)).doubleValue();
        wpos_right_tempered = ((Double) vin.elementAt(12)).doubleValue();
        wneg_right_tempered = ((Double) vin.elementAt(13)).doubleValue();

        wpos_left_codensity = ((Double) vin.elementAt(14)).doubleValue();
        wneg_left_codensity = ((Double) vin.elementAt(15)).doubleValue();
        wpos_right_codensity = ((Double) vin.elementAt(16)).doubleValue();
        wneg_right_codensity = ((Double) vin.elementAt(17)).doubleValue();

        number_nodes++;
        leftnn =
            new DecisionTreeNode(
                this,
                number_nodes,
                nn.depth + 1,
                split_CV,
                (Vector) vin.elementAt(4),
                pos_left,
                neg_left,
                wpos_left_tempered,
                wneg_left_tempered,
                wpos_left_codensity,
                wneg_left_codensity);

        leftnn.compute_prediction(t_leaf);

        // System.out.println("left_leaf : " + t_leaf + ", wpos_left_codensity = " +
        // wpos_left_codensity + ", wneg_left_codensity = " + wneg_left_codensity);

        leftnn.continuous_features_indexes_for_split_copy_from(nn);
        leftnn.continuous_features_indexes_for_split_update_child(
            nn.feature_node_index, nn.feature_node_test_index, DecisionTreeNode.LEFT_CHILD);

        DecisionTree.INSERT_WEIGHT_ORDER(try_leaves, leftnn);

        number_nodes++;
        rightnn =
            new DecisionTreeNode(
                this,
                number_nodes,
                nn.depth + 1,
                split_CV,
                (Vector) vin.elementAt(5),
                pos_right,
                neg_right,
                wpos_right_tempered,
                wneg_right_tempered,
                wpos_right_codensity,
                wneg_right_codensity);

        rightnn.compute_prediction(t_leaf);

        // System.out.println("right_leaf : " + t_leaf + ", wpos_right_codensity = " +
        // wpos_right_codensity + ", wneg_right_codensity = " + wneg_right_codensity);

        rightnn.continuous_features_indexes_for_split_copy_from(nn);
        rightnn.continuous_features_indexes_for_split_update_child(
            nn.feature_node_index, nn.feature_node_test_index, DecisionTreeNode.RIGHT_CHILD);

        DecisionTree.INSERT_WEIGHT_ORDER(try_leaves, rightnn);

        if (nn.depth + 1 > depth) depth = nn.depth + 1;

        nn.left_child = leftnn;
        nn.right_child = rightnn;

        nsplits++;
      }
      if (number_nodes >= max_size) stop = true;

    } while (!stop);

    // updates leaves in tree
    leaves = new Vector();
    for (j = 0; j < try_leaves.size(); j++) leaves.addElement(try_leaves.elementAt(j));
  }

  public void init(double tt) {
    int i, ne = myBoost.myDomain.myDS.train_size(split_CV);
    Vector indexes = new Vector();
    Example e;
    int pos = 0, neg = 0;
    double wpos_tempered = 0.0,
        wneg_tempered = 0.0,
        wpos_codensity = 0.0,
        wneg_codensity = 0.0,
        alpha_leaf;

    for (i = 0; i < ne; i++) {
      indexes.addElement(new Integer(i));
      e = myBoost.myDomain.myDS.train_example(split_CV, i);
      if (e.is_positive_noisy()) {
        pos++;
        wpos_tempered += e.current_boosting_weight;
        if (myBoost.name.equals(Boost.KEY_NAME_TEMPERED_LOSS))
          wpos_codensity += Math.pow(e.current_boosting_weight, 2.0 - tt);
        else wpos_codensity = -1.0; // enforces checkable errors
      } else {
        neg++;
        wneg_tempered += e.current_boosting_weight;
        if (myBoost.name.equals(Boost.KEY_NAME_TEMPERED_LOSS))
          wneg_codensity += Math.pow(e.current_boosting_weight, 2.0 - tt);
        else wpos_codensity = -1.0; // enforces checkable errors
      }
    }

    number_nodes = 1;

    root =
        new DecisionTreeNode(
            this,
            number_nodes,
            0,
            split_CV,
            indexes,
            pos,
            neg,
            wpos_tempered,
            wneg_tempered,
            wpos_codensity,
            wneg_codensity);
    root.init_continuous_features_indexes_for_split();
    depth = 0;

    // System.out.println("tt = " + tt + ", root_leaf : " + root + ", wpos_codensity = " +
    // wpos_codensity + ", wneg_codensity = " + wneg_codensity);

    root.compute_prediction(myBoost.tempered_t);

    leaves = new Vector();
    leaves.addElement(root);
  }

  public DecisionTreeNode get_leaf(Example ee) {
    // returns the leaf reached by the example
    DecisionTreeNode nn = root;
    Feature f;
    while (!nn.is_leaf) {
      f =
          (Feature)
              myBoost.myDomain.myDS.features.elementAt(
                  myBoost
                      .myDomain
                      .myDS
                      .index_observation_features_to_index_features[nn.feature_node_index]);
      if (f.example_goes_left(ee, nn.feature_node_index, nn.feature_node_test_index))
        nn = nn.left_child;
      else nn = nn.right_child;
    }
    return nn;
  }

  public DecisionTreeNode get_leaf_MonotonicTreeGraph(Example ee) {
    // returns the monotonic node reached by the example
    // (builds a strictly monotonic path of nodes to a leaf, used the last one in the path; this is
    // a prediction node in the corresponding MonotonicTreeGraph)

    DecisionTreeNode nn = root;
    double best_prediction = Math.abs(nn.node_prediction_from_boosting_weights);
    DecisionTreeNode ret = root;

    Feature f;
    while (!nn.is_leaf) {
      if (Math.abs(nn.node_prediction_from_boosting_weights) > best_prediction) {
        best_prediction = Math.abs(nn.node_prediction_from_boosting_weights);
        ret = nn;
      }

      f =
          (Feature)
              myBoost.myDomain.myDS.features.elementAt(
                  myBoost
                      .myDomain
                      .myDS
                      .index_observation_features_to_index_features[nn.feature_node_index]);
      if (f.example_goes_left(ee, nn.feature_node_index, nn.feature_node_test_index))
        nn = nn.left_child;
      else nn = nn.right_child;
    }

    return ret;
  }

  public double leveraging_mu() {
    if (myBoost.name.equals(Boost.KEY_NAME_TEMPERED_LOSS)) return leveraging_mu_tempered_loss();
    else if (myBoost.name.equals(Boost.KEY_NAME_LOG_LOSS)) return leveraging_mu_log_loss();
    else Dataset.perror("DecisionTree.class :: no loss " + myBoost.name);

    return -1.0;
  }

  public double leveraging_mu_log_loss() {
    int i, ne = myBoost.myDomain.myDS.train_size(split_CV);
    double rho_j = 0.0, max_absolute_pred = -1.0, output_e, tot_weight = 0.0, mu_j;
    Example e;

    for (i = 0; i < ne; i++) {
      e = myBoost.myDomain.myDS.train_example(split_CV, i);
      output_e = output_boosting(e);
      if ((i == 0) || (Math.abs(output_e) > max_absolute_pred))
        max_absolute_pred = Math.abs(output_e);

      rho_j += (e.current_boosting_weight * output_e * e.noisy_normalized_class);
      tot_weight += e.current_boosting_weight;
    }
    rho_j /= (tot_weight * max_absolute_pred);

    tree_p_t = (1.0 + rho_j) / 2.0;
    tree_psi_t = Math.log((1.0 + rho_j) / (1.0 - rho_j));

    tree_p_t_star = 1.0 / (1.0 + Math.exp(-max_absolute_pred));

    if (tree_psi_t < 0.0)
      Dataset.perror("DecisionTree.class :: negative value tree_psi_t = " + tree_psi_t);

    // if ( (rho_j >= 1.0) || (rho_j <= -1.0) )
    //  Dataset.perror("DecisionTree.class :: impossible value rho_j = " + rho_j); REMOVE

    mu_j = (Math.log((1.0 + rho_j) / (1.0 - rho_j))) / max_absolute_pred;
    return mu_j;
  }

  public double leveraging_mu_tempered_loss() {

    int i, ne = myBoost.myDomain.myDS.train_size(split_CV), maxr, nbpos = 0, nbnull = 0;
    double r_j = 0.0, epsilon_j = 0.0, rho_j = 0.0, ww, mu_j, rat;
    Example e;

    // computes r_j, see paper

    for (i = 0; i < ne; i++) {
      e = myBoost.myDomain.myDS.train_example(split_CV, i);
      if (e.current_boosting_weight < 0.0)
        Dataset.perror("DecisionTree.class :: example has negative weight");

      if (e.current_boosting_weight > 0.0) {
        if (myBoost.tempered_t != 1.0)
          rat =
              Math.abs(unweighted_edge_training(e))
                  / Math.pow(e.current_boosting_weight, 1.0 - myBoost.tempered_t);
        else rat = Math.abs(unweighted_edge_training(e));

        if ((nbpos == 0) || (rat > r_j)) r_j = rat;

        nbpos++;
      }
    }

    if (r_j == 0.0) // h = 0 => mu = 0
    return 0.0;

    // computes epsilon_j, see paper

    if (myBoost.tempered_t != 1.0) {
      for (i = 0; i < ne; i++) {
        e = myBoost.myDomain.myDS.train_example(split_CV, i);
        if (e.current_boosting_weight == 0.0) {
          rat = Math.abs(unweighted_edge_training(e)) / r_j;

          if ((nbnull == 0) || (rat > epsilon_j)) {
            epsilon_j = rat;
          }

          nbnull++;
        }
      }
      if (nbnull > 0) epsilon_j = Math.pow(epsilon_j, 1.0 / (1.0 - myBoost.tempered_t));
    }

    // computes rho_j, see paper

    for (i = 0; i < ne; i++) {
      e = myBoost.myDomain.myDS.train_example(split_CV, i);
      ww = (e.current_boosting_weight > 0.0) ? e.current_boosting_weight : epsilon_j;
      rho_j += ww * unweighted_edge_training(e);
    }

    rho_j /= ((1.0 + (((double) nbnull) * Math.pow(epsilon_j, 2.0 - myBoost.tempered_t))) * r_j);

    if ((rho_j > 1.0) || (rho_j < -1.0))
      Dataset.perror("DecisionTree.class :: rho_j = " + rho_j + " not in [-1,1]");

    // System.out.println("BEFORE " + myBoost.tempered_t + ", rho = " + rho_j);

    mu_j =
        -Statistics.TEMPERED_LOG(
            (1.0 - rho_j) / Statistics.Q_MEAN(1.0 - rho_j, 1.0 + rho_j, 1.0 - myBoost.tempered_t),
            myBoost.tempered_t);

    // System.out.println("AFTER " + myBoost.tempered_t + ", mu = " + mu_j);

    if ((Double.isNaN(mu_j)) || (Double.isInfinite(mu_j))) {
      if (rho_j > 0.0) mu_j = Boost.MAX_PRED_VALUE;
      else if (rho_j < 0.0) mu_j = -Boost.MAX_PRED_VALUE;
      else Dataset.perror("DecisionTree.class :: inconsistency in the computation of rho_j");

      TemperedBoostException.ADD(TemperedBoostException.NUMERICAL_ISSUES_INFINITE_MU);
    }

    mu_j /= r_j;

    return mu_j;
  }

  public double leveraging_alpha(double mu, Vector<Double> allzs) {
    if ((myBoost.tempered_t == 1.0) || (!myBoost.name.equals(Boost.KEY_NAME_TEMPERED_LOSS)))
      return mu;

    double al =
        Math.pow(
            (double) myBoost.myDomain.myDS.train_size(split_CV),
            1.0 - Statistics.STAR(myBoost.tempered_t));
    int i;
    if (name > 0) {
      for (i = 0; i < name; i++)
        al *= Math.pow(allzs.elementAt(i).doubleValue(), 1.0 - myBoost.tempered_t);
    }
    al *= mu;

    return al;
  }

  public double output_boosting(Example ee) {
    DecisionTreeNode nn = get_leaf(ee);

    // if (ee.domain_id == 1){
    //  System.out.println("DT [" + nn.name + "," + nn.depth + "," +
    // nn.node_prediction_from_boosting_weights + "]"); // REMOVE
    // }

    nn.checkForOutput();

    return (nn.node_prediction_from_boosting_weights);
  }

  public double output_boosting_MonotonicTreeGraph(Example ee) {
    DecisionTreeNode nn = get_leaf_MonotonicTreeGraph(ee);

    // if (ee.domain_id == 1){
    //  System.out.println("MTG [" + nn.name + "," + nn.depth + "," +
    // nn.node_prediction_from_boosting_weights + "]"); // REMOVE
    // }

    nn.checkForOutput_MonotonicTreeGraph();

    return (nn.node_prediction_from_boosting_weights);
  }

  public double unweighted_edge_training(Example ee) {
    // return y * this(ee) ; USE NOISY CLASS (if no noise, just the regular class)
    return ((output_boosting(ee)) * (ee.noisy_normalized_class));
  }

  public double unweighted_edge_training_MonotonicTreeGraph(Example ee) {
    // return y * this(ee) ; USE NOISY CLASS (if no noise, just the regular class)
    return ((output_boosting_MonotonicTreeGraph(ee)) * (ee.noisy_normalized_class));
  }

  // safe checks methods

  public void check() {
    int i, j, ne = myBoost.myDomain.myDS.train_size(split_CV);
    DecisionTreeNode leaf1, leaf2;
    Example ee;

    for (i = 0; i < leaves.size(); i++) {
      leaf1 = (DecisionTreeNode) leaves.elementAt(i);
      for (j = 0; j < leaf1.train_fold_indexes_in_node.length; j++) {
        ee = myBoost.myDomain.myDS.train_example(split_CV, leaf1.train_fold_indexes_in_node[j]);

        leaf2 = get_leaf(ee);
        if (leaf1.name != leaf2.name) {
          Dataset.perror(
              "DecisionTree.class :: Example "
                  + ee
                  + " reaches leaf #"
                  + leaf2
                  + " but is recorded for leaf #"
                  + leaf1);
        }
      }
    }
  }
}
