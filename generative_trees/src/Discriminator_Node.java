// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Discriminator_Node
 *****/

class Discriminator_Node implements Debuggable {
  int name, depth, split_CV;
  // index of the node in the tree, the root is labeled 0; depth = depth in the tree (0 = root);
  // split_CV = # fold for training

  Discriminator_Node left_child, right_child;
  boolean is_leaf;
  // if true, left_child = right_child = null;

  boolean
      leaf_tested_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values;
  boolean
      leaf_value_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values;

  double link_value, local_p;
  // real valued prediction, to be used only for leaves; local_p = relative weight of positives in
  // leaf

  int[] train_fold_indexes_in_node_real;
  int[] train_fold_indexes_in_node_fake;
  // index of examples in the training fold that reach the node

  int n_real, n_fake;
  // #pos and neg examples in the training fold reaching the node (CARDINALS)

  double w_real, w_fake;
  // sum of weights of pos and neg examples in the training fold reaching the node

  boolean computed;
  // true iff the node has been fully computed as internal (left and right children) or leaf

  // Feature information at the node level
  Discriminator_Tree myDiscriminatorTree;

  int feature_node_index;
  // handle on the feature index
  int feature_node_test_index;
  // index of the feature test in the feature tests vector
  // =tie for a test on continuous or integer values ( <= is left, > is right)
  // =tie for a test on nominal values ( in the set is left, otherwise is right)

  Vector<Feature> support_at_classification_node;
  // keeps the complete support at given node -- useful for the Chi2 & copycat approach

  public static Vector SURROGATE_split_characteristics(Vector surrogate_split) {
    // sopies everything BUT the node to save space;

    Vector<Integer> left_real =
        new Vector<>((Vector<Integer>) ((Vector) surrogate_split.elementAt(4)).elementAt(0));
    Vector<Integer> right_real =
        new Vector<>((Vector<Integer>) ((Vector) surrogate_split.elementAt(5)).elementAt(0));
    Vector<Integer> left_fake =
        new Vector<>((Vector<Integer>) ((Vector) surrogate_split.elementAt(4)).elementAt(1));
    Vector<Integer> right_fake =
        new Vector<>((Vector<Integer>) ((Vector) surrogate_split.elementAt(5)).elementAt(1));

    Vector<Vector> left = new Vector<>();
    left.addElement(left_real);
    left.addElement(left_fake);

    Vector<Vector> right = new Vector<>();
    right.addElement(right_real);
    right.addElement(right_fake);

    Vector v_inv = new Vector();

    v_inv.addElement(new Double(((Double) surrogate_split.elementAt(0)).doubleValue())); // 0
    v_inv.addElement(null); // 1
    v_inv.addElement(new Integer(((Integer) surrogate_split.elementAt(2)).intValue())); // 2
    v_inv.addElement(new Integer(((Integer) surrogate_split.elementAt(3)).intValue())); // 3
    v_inv.addElement(left); // 4
    v_inv.addElement(right); // 5
    v_inv.addElement(new Integer(((Integer) surrogate_split.elementAt(6)).intValue())); // 6
    v_inv.addElement(new Integer(((Integer) surrogate_split.elementAt(7)).intValue())); // 7
    v_inv.addElement(new Integer(((Integer) surrogate_split.elementAt(8)).intValue())); // 8
    v_inv.addElement(new Integer(((Integer) surrogate_split.elementAt(9)).intValue())); // 9
    v_inv.addElement(new Double(((Double) surrogate_split.elementAt(10)).doubleValue())); // 10
    v_inv.addElement(new Double(((Double) surrogate_split.elementAt(11)).doubleValue())); // 11
    v_inv.addElement(new Double(((Double) surrogate_split.elementAt(12)).doubleValue())); // 12
    v_inv.addElement(new Double(((Double) surrogate_split.elementAt(13)).doubleValue())); // 13

    return v_inv;
  }

  Discriminator_Node() {
    name = -1;
    computed =
        leaf_tested_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values =
            leaf_value_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values =
                false;

    left_child = right_child = null;
    train_fold_indexes_in_node_real = train_fold_indexes_in_node_fake = null;
    is_leaf = true;

    link_value = 0.0;

    feature_node_index = -1;
    feature_node_test_index = -1;

    support_at_classification_node = null;
  }

  Discriminator_Node(Discriminator_Tree t, int v, int d) {
    this();
    myDiscriminatorTree = t;
    name = v;
    depth = d;
  }

  Discriminator_Node(
      Discriminator_Tree t,
      int v,
      int d,
      int split,
      Vector indexes_real,
      Vector indexes_fake,
      int p,
      int n,
      double wp,
      double wn,
      Vector<Feature> sn) {
    this(t, v, d);
    split_CV = split;

    if (sn == null) Dataset.perror("Discriminator_Node.class :: no support given for new node");

    support_at_classification_node = sn;

    int i;
    if (indexes_real != null) {
      train_fold_indexes_in_node_real = new int[indexes_real.size()];
      for (i = 0; i < indexes_real.size(); i++) {
        train_fold_indexes_in_node_real[i] = ((Integer) indexes_real.elementAt(i)).intValue();
      }
    } else train_fold_indexes_in_node_real = null;

    if (indexes_fake != null) {
      train_fold_indexes_in_node_fake = new int[indexes_fake.size()];
      for (i = 0; i < indexes_fake.size(); i++) {
        train_fold_indexes_in_node_fake[i] = ((Integer) indexes_fake.elementAt(i)).intValue();
      }
    } else train_fold_indexes_in_node_fake = null;

    n_real = p;
    n_fake = n;

    w_real = wp;
    w_fake = wn;

    if (((train_fold_indexes_in_node_real == null) && (n_real > 0))
        || (train_fold_indexes_in_node_real.length != n_real))
      Dataset.perror(
          "Discriminator_Node.class :: attempting to create a node with mismatch in #reaching real"
              + " examples");

    if (((train_fold_indexes_in_node_fake == null) && (n_fake > 0))
        || (train_fold_indexes_in_node_fake.length != n_fake))
      Dataset.perror(
          "Discriminator_Node.class :: attempting to create a node with mismatch in #reaching fake"
              + " examples");
  }

  public boolean equals(Object o) {
    if (o == this) return true;
    if (!(o instanceof Discriminator_Node)) return false;
    Discriminator_Node test = (Discriminator_Node) o;
    if ((test.name == name)
        && (test.myDiscriminatorTree.name == myDiscriminatorTree.name)
        && (test.depth == depth)) return true;
    return false;
  }

  public double total_weight(boolean use_weights) {
    if (use_weights) return (w_real + w_fake);
    else return ((double) n_real + n_fake);
  }

  public boolean is_pure() {
    if ((n_real == 0) || (w_real == 0.0) || (n_fake == 0) || (w_fake == 0.0)) {
      if (((n_real == 0) && (w_real != 0.0)) || ((n_real != 0) && (w_real == 0.0)))
        Dataset.perror("Discriminator.class :: issue with positive counting at node " + this);
      if (((n_fake == 0) && (w_fake != 0.0)) || ((n_fake != 0) && (w_fake == 0.0)))
        Dataset.perror("Discriminator.class :: issue with negative counting at node " + this);
      return true;
    }
    return false;
  }

  public void compute_p() {
    if (w_real + w_fake == 0.0)
      Dataset.perror("Discriminator_Node.class :: wpos + wneg == 0.0 w/o DP");

    local_p = Discriminator_Tree.COMPUTE_P(w_real, w_fake);
  }

  public void compute_link(double alpha) {
    if ((alpha < 0.0) || (alpha > 1.0))
      Dataset.perror(
          "Discriminator_Node.class compute_link :: alpha (" + alpha + ") should be in [0,1]");

    compute_p();
    if (w_real + w_fake != 0.0) link_value = Statistics.CANONICAL_LINK_MATUSITA(alpha, local_p);
    else {
      if (local_p < 0.5) link_value = -EPS;
      else if (local_p > 0.5) link_value = EPS;
      else Dataset.perror("Discriminator_Node.class :: link for node #" + name + " == 0.0");
    }

    computed = true;
  }

  public String binary_decision() {
    if (link_value == 0)
      Dataset.perror("Discriminator_Node.class :: link for node #" + name + " == 0.0");
    else if (!is_leaf) Dataset.perror("Discriminator_Node.class :: not a leaf");

    if (link_value > 0.0) return "REAL";
    else return "FAKE";
  }

  public static Vector REAL_OR_FAKE_VECTOR(
      Vector<Integer> vector_real, Vector<Integer> vector_fake, boolean use_real) {
    if (use_real) return vector_real;
    else return vector_fake;
  }

  public int[] train_fold_indexes_in_node(boolean use_real) {
    if (use_real) return train_fold_indexes_in_node_real;
    else return train_fold_indexes_in_node_fake;
  }

  public void add_argument(
      int value, Vector<Integer> vector_real, Vector<Integer> vector_fake, boolean use_real) {
    if (use_real) vector_real.addElement(new Integer(value));
    else vector_fake.addElement(new Integer(value));
  }

  public boolean
      has_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values(
          boolean use_weights) {
    if (leaf_tested_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values)
      return leaf_value_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values; // FASTER

    int i, j, k, n_real_left, n_real_right;
    Feature f;
    Example e;
    double w_real_left, w_real_right;

    for (i = 0; i < myDiscriminatorTree.myBoost.myDomain.myDS.number_domain_features(); i++) {
      f = support_at_classification_node.elementAt(i);

      if ((!Feature.HAS_SINGLETON_DOMAIN(f)) && (Feature.SPLIT_AUTHORIZED(f))) {
        for (j = 0; j < f.tests.size(); j++) {
          w_real_left = w_real_right = 0.0;
          n_real_left = n_real_right = 0;
          for (k = 0; k < train_fold_indexes_in_node(true).length; k++) {
            e =
                myDiscriminatorTree.myBoost.myDomain.myDS.train_example(
                    split_CV, train_fold_indexes_in_node(true)[k], true);
            if ((!Example.FEATURE_IS_UNKNOWN(e, i)) && (f.example_goes_left(e, i, j, true)))
              if (use_weights) w_real_left += e.unnormalized_weight;
              else n_real_left++;
            else if (!Example.FEATURE_IS_UNKNOWN(e, i))
              if (use_weights) w_real_right += e.unnormalized_weight;
              else n_real_right++;

            if (((use_weights) && (w_real_left > 0.0) && (w_real_right > 0.0))
                || ((!use_weights) && (n_real_left > 0) && (n_real_right > 0))) {
              leaf_tested_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values =
                  true;
              leaf_value_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values =
                  true;
              return true;
            }
          }
        }
      }
    }
    leaf_tested_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values =
        true;
    leaf_value_for_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values =
        false;
    return false;
  }

  public Vector split_characteristics(
      int feature_index, int feature_test_index, double alpha, boolean relative) {
    int k, n_real_left, n_real_right, n_fake_left, n_fake_right, tot_left, tot_right, iindex, itype;
    double delta, w_real_left, w_real_right, w_fake_left, w_fake_right, cweight, toss;

    Vector v_inv = null;
    Feature f, f_left, f_right;
    f = support_at_classification_node.elementAt(feature_index);

    if (Feature.HAS_SINGLETON_DOMAIN(f)) return null;
    Vector<Vector> left, right;

    Vector<Integer> left_real = new Vector<>();
    Vector<Integer> right_real = new Vector<>();
    Vector<Integer> left_fake = new Vector<>();
    Vector<Integer> right_fake = new Vector<>();
    Example e;
    boolean use_real;

    v_inv = new Vector();

    n_real_left = n_real_right = n_fake_left = n_fake_right = tot_left = tot_right = 0;
    w_real_left = w_real_right = w_fake_left = w_fake_right = 0.0;

    for (itype = 1; itype >= 0; itype--) {
      // 0 : fake, 1 = real
      if (itype == 0) use_real = false;
      else use_real = true;

      for (k = 0; k < train_fold_indexes_in_node(use_real).length; k++) {
        e =
            myDiscriminatorTree.myBoost.myDomain.myDS.train_example(
                split_CV, train_fold_indexes_in_node(use_real)[k], use_real);
        toss = Algorithm.RANDOM_P_NOT_HALF();
        if (f.example_goes_left(e, feature_index, feature_test_index, true)) {

          add_argument(train_fold_indexes_in_node(use_real)[k], left_real, left_fake, use_real);
          tot_left++;
          if (e.is_positive()) {
            w_real_left += e.unnormalized_weight;
            n_real_left++;
          } else {
            w_fake_left += e.unnormalized_weight;
            n_fake_left++;
          }
        } else {
          // example goes right

          add_argument(train_fold_indexes_in_node(use_real)[k], right_real, right_fake, use_real);
          tot_right++;
          if (e.is_positive()) {
            w_real_right += e.unnormalized_weight;
            n_real_right++;
          } else {
            w_fake_right += e.unnormalized_weight;
            n_fake_right++;
          }
        }
      }
    }

    delta =
        Statistics.DELTA_PHI_SPLIT(
            alpha, w_real, w_fake, w_real_left, w_fake_left, w_real_right, w_fake_right, relative);

    if (delta < 0.0)
      Dataset.perror(
          " delta "
              + delta
              + " < 0 -- alpha = "
              + alpha
              + ", w_real = "
              + w_real
              + ", w_fake = "
              + w_fake
              + ", w_real_left = "
              + w_real_left
              + ", w_fake_left = "
              + w_fake_left
              + ", w_real_right = "
              + w_real_right
              + ", w_fake_right = "
              + w_fake_right);

    left = new Vector<>();
    left.addElement(left_real);
    left.addElement(left_fake);

    right = new Vector<>();
    right.addElement(right_real);
    right.addElement(right_fake);

    v_inv.addElement(new Double(delta)); // 0
    v_inv.addElement(this); // 1
    v_inv.addElement(new Integer(feature_index)); // 2
    v_inv.addElement(new Integer(feature_test_index)); // 3
    v_inv.addElement(left); // 4
    v_inv.addElement(right); // 5
    v_inv.addElement(new Integer(n_real_left)); // 6
    v_inv.addElement(new Integer(n_fake_left)); // 7
    v_inv.addElement(new Integer(n_real_right)); // 8
    v_inv.addElement(new Integer(n_fake_right)); // 9
    v_inv.addElement(new Double(w_real_left)); // 10
    v_inv.addElement(new Double(w_fake_left)); // 11
    v_inv.addElement(new Double(w_real_right)); // 12
    v_inv.addElement(new Double(w_fake_right)); // 13

    return v_inv;
  }

  public boolean split_is_eligible(Vector split_characteristics) {
    // uses data structure of a split in split_characteristics and returns true iff split can be
    // used as split

    if (Boost.AUTHORISE_REAL_PURE_LEAVES) return true;

    int posl, posr;
    posl = ((Vector) ((Vector) split_characteristics.elementAt(4)).elementAt(0)).size();
    posr = ((Vector) ((Vector) split_characteristics.elementAt(5)).elementAt(0)).size();

    if ((posl > 0) && (posr > 0)) return true;
    return false;
  }

  public Vector allSplits(double alpha, boolean relative) {
    Vector<Vector> v = new Vector<>();

    int i, j;
    Feature f;

    for (i = 0; i < myDiscriminatorTree.myBoost.myDomain.myDS.number_domain_features(); i++) {
      f = support_at_classification_node.elementAt(i);

      if ((!Feature.HAS_SINGLETON_DOMAIN(f)) && (Feature.SPLIT_AUTHORIZED(f))) {
        for (j = 0; j < f.tests.size(); j++)
          v.addElement(split_characteristics(i, j, alpha, relative));
      }
    }
    return v;
  }

  public String toString() {
    String v = "";
    int leftn, rightn;

    if (name != 1) v += "[#" + name + "]";
    else v += "[#1:root]";
    if ((is_leaf) && (w_real == 0.0) && (w_fake == 0.0))
      v += " leaf+- (" + DF4.format(link_value) + ")";
    else if ((is_leaf) && (w_real == 0.0)) v += " leaf- (" + DF4.format(link_value) + ")";
    else if ((is_leaf) && (w_fake == 0.0)) v += " leaf+ (" + DF4.format(link_value) + ")";
    else if (is_leaf) v += " leaf* (" + DF4.format(link_value) + ")";
    else {
      if ((left_child != null) && (right_child != null))
        v +=
            " internal ("
                + support_at_classification_node
                    .elementAt(feature_node_index)
                    .display_test(feature_node_test_index)
                + " ? #"
                + left_child.name
                + " : #"
                + right_child.name
                + ")";
      else {
        if (left_child != null) leftn = left_child.name;
        else leftn = -1;

        if (right_child != null) rightn = right_child.name;
        else rightn = -1;

        v +=
            " internal ("
                + support_at_classification_node
                    .elementAt(feature_node_index)
                    .display_test(feature_node_test_index)
                + " ? #"
                + leftn
                + " : #"
                + rightn
                + ")";
      }
    }

    v +=
        " {" + n_real + "|" + n_fake + "} {" + total_weight(true) + "|" + total_weight(false) + "}";

    v += "\n";

    return v;
  }

  public String display(HashSet indexes) {
    String v = "";
    int i;
    HashSet dum;
    boolean bdum;

    for (i = 0; i < depth; i++) {
      if ((i == depth - 1) && (indexes.contains(new Integer(i)))) v += "|-";
      else if (i == depth - 1) v += "\\-";
      else if (indexes.contains(new Integer(i))) v += "| ";
      else v += "  ";
    }
    v += toString();

    if (!is_leaf) {
      dum = new HashSet(indexes);
      bdum = dum.add(new Integer(depth));

      if (left_child != null) v += left_child.display(dum);
      else v += "null";

      if (right_child != null) v += right_child.display(indexes);
      else v += "null";
    }

    return v;
  }

  public String toStringInDiscriminatorTree() {
    String v = "";
    int i;

    for (i = 0; i < depth; i++) {
      if (i == 0) v += ">";
      else v += "-";
    }
    v += toString();

    return v;
  }
}
