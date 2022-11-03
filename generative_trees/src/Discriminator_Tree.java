// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Discriminator_Tree
 *****/

class Discriminator_Tree implements Debuggable {

  // variables to speed up the induction

  public static boolean RANDOMISE_SPLIT_FINDING_WHEN_TOO_MANY_SPLITS = true;
  public static int MAX_SPLITS_BEFORE_RANDOMISATION = 1000;
  public static int MAX_CARD_MODALITIES_BEFORE_RANDOMISATION =
      10; // for NOMINAL variables: partially computes the number of candidates splits if modalities
  // > MAX_CARD_MODALITIES_BEFORE_RANDOMISATION
  public static int MAX_SIZE_FOR_RANDOMISATION =
      2; // used to partially compute the number of candidates splits for NOMINAL variables (here,
  // O({n\choose MAX_SIZE_FOR_RANDOMISATION}))

  // combinatorial split design variables

  public static boolean USE_OBSERVED_FEATURE_VALUES_FOR_SPLITS = false;

  int name, depth;

  Discriminator_Node root;
  // root of the tree

  Vector<Discriminator_Node> leaves;
  // list of leaves of the tree (potential growth here)
  Hashtable<Discriminator_Node, Integer> leaf_to_index_in_leaves;
  // maps leaves to their index in leaves;

  Boost myBoost;

  int max_depth, split_CV;

  int number_nodes;
  // includes leaves

  double unnormalized_weights_sum;

  public static double COMPUTE_P(double wp, double wn) {
    // first implementation: computes just the result
    double val;

    if (wp + wn > 0.0) val = (wp / (wp + wn));
    else val = 0.5;

    if (val != 0.5) return val;

    // p = 0.5: makes a slight random shift to ensure non-zero canonical link
    double vv = Algorithm.RANDOM_P_NOT_HALF();

    if (vv < 0.5) val -= EPS2;
    else val += EPS2;
    return val;
  }

  public static boolean SPLIT(Vector v) {
    // controls leaves would not be pure

    if ((((Vector) v.elementAt(4)).size() == 0) || (((Vector) v.elementAt(5)).size() == 0))
      return false;

    return true;
  }

  Discriminator_Tree(int nn, Boost bb, int maxs) {
    name = nn;
    root = null;
    myBoost = bb;

    max_depth = maxs;

    unnormalized_weights_sum = 0.0;

    leaf_to_index_in_leaves = null;
  }

  Discriminator_Tree(int nn, Boost bb, int maxs, int split) {
    this(nn, bb, maxs);

    split_CV = split;
  }

  public void init_real_examples_training_leaf(Discriminator_Node dn_init) {
    int i;
    for (i = 0; i < myBoost.myDomain.myDS.train_size(split_CV, true); i++) {
      myBoost.myDomain.myDS.train_example(split_CV, i, true).update_training_leaf(dn_init);
    }
  }

  public String toString() {
    int i;

    String v = "(name = #" + name + " | depth = " + depth + " | #nodes = " + number_nodes + ")\n";
    boolean stop = false;

    Discriminator_Node n = root;
    Vector<Discriminator_Node> cur_nodes = new Vector<>();
    cur_nodes.addElement(n);

    Vector<Discriminator_Node> next_nodes;

    v += root.display(new HashSet());

    v += "Decision Leaves: ";
    for (i = 0; i < leaves.size(); i++) {
      v +=
          "(#"
              + ((Discriminator_Node) leaves.elementAt(i)).name
              + ":"
              + ((Discriminator_Node) leaves.elementAt(i)).binary_decision()
              + ")";
      if (i < leaves.size() - 1) v += " ";
    }
    v += ".\n";

    return v;
  }

  public void compute_leaves_hashtable() {
    if ((leaves == null) || (leaves.size() == 0))
      Dataset.perror("Discriminator_Tree.class :: no leaves to hash");

    leaf_to_index_in_leaves = new Hashtable<Discriminator_Node, Integer>();
    int i;
    for (i = 0; i < leaves.size(); i++)
      if (!((Discriminator_Node) leaves.elementAt(i)).is_leaf)
        Dataset.perror("Discriminator_Tree.class :: found a non-leaf node in leaves");
      else leaf_to_index_in_leaves.put((Discriminator_Node) leaves.elementAt(i), new Integer(i));
  }

  public int heaviest_leaf_in(
      Vector<Discriminator_Node> try_leaves, Vector<Integer> indexes, int max_depth)
      throws NoLeafFoundException {
    // returns the index in indexes of the heaviest (total example weight) leaf in try_leaves among
    // those with depth <= max_depth

    int i, j, heaviest_leaf_index = -1, ntested = 0;
    double cbrbest = -1.0;
    Discriminator_Node nn;
    boolean one_potential_heaviest_leaf = false;

    for (i = 0; i < indexes.size(); i++) {
      j = indexes.elementAt(i).intValue();
      nn = try_leaves.elementAt(j);

      if ((nn.depth < max_depth)
          && (nn
              .has_one_split_with_non_positive_pure_leaves_for_completely_specified_attribute_values(
                  true))) {
        if ((ntested == 0) || (nn.total_weight(true) > cbrbest)) {
          cbrbest = nn.total_weight(true);
          heaviest_leaf_index = j;
          one_potential_heaviest_leaf = true;
          ntested++;
        }
      } else Algorithm.FEAT_SIZE_2++;
    }

    if (!one_potential_heaviest_leaf) throw new NoLeafFoundException("No leaf for indexes");

    return heaviest_leaf_index;
  }

  public Vector output_vector_grow(String s, Discriminator_Node dn) {
    Vector vret = new Vector();
    vret.addElement(s);
    vret.addElement(dn);
    return vret;
  }

  public Vector one_step_grow_cheaper() {
    if (number_nodes > myBoost.max_size_discriminator_tree)
      Dataset.perror(
          "Discriminator_tree.class :: tree size "
              + number_nodes
              + " larger than max size = "
              + myBoost.max_size_discriminator_tree);
    if (number_nodes == myBoost.max_size_discriminator_tree)
      return output_vector_grow(Algorithm.NO_DT_SPLIT_MAX_SIZE, null);

    Vector vin = null;
    boolean stop = false, check_contain, found_leaf;
    Discriminator_Node nn, leftnn, rightnn, nnret = null;

    int i,
        j,
        i_split,
        j_split,
        j_split_index,
        k,
        ll,
        ibest = -1,
        pos_left,
        neg_left,
        pos_right,
        neg_right,
        test_leaf_index = -1;
    long n_tests;

    double wpos_left, wneg_left, wpos_right, wneg_right, alpha_leaf, dum_err, deltacur, deltabest;

    Feature f_split;

    Vector<Discriminator_Node> new_leaves = null;
    Vector<Integer> pure_leaves = null;
    Vector<Integer> split_leaves_with_no_impure_split = null;

    Vector<Discriminator_Node> stay_leaves = null;
    Vector cur_splits;

    Vector<Vector> left, right;
    Vector<Integer> dummy_v;
    Example e;

    Vector<Feature>[] split_supports;

    Vector<Integer> try_leaves_indexes = new Vector<>();
    Vector split_characteristics, split_characteristics_best;

    int size_verbose_tests = 100000;
    int size_verbose_train = 10000;

    for (j = 0; j < leaves.size(); j++) try_leaves_indexes.addElement(new Integer(j));

    if ((try_leaves_indexes == null) || (try_leaves_indexes.size() == 0))
      Dataset.perror("Discriminator_Tree.class :: no leaf");

    alpha_leaf = myBoost.getAlpha(this);

    Algorithm.FEAT_SIZE = 0;
    if (MAX_SPLITS_BEFORE_RANDOMISATION < Feature.NUMBER_CONTINUOUS_TIES)
      Dataset.perror("Discriminator_Tree.class :: not enough tests for 'big' features");

    Algorithm.CHECK_STRATEGY_DT_GROW_CONTAINS(myBoost.strategy_dt_grow_one_leaf);
    do {
      do {
        found_leaf = true;
        Algorithm.FEAT_SIZE_2 = 0; // counts # leaves with no fully unpure split
        try {
          if (myBoost.strategy_dt_grow_one_leaf.equals(
              Algorithm.STRATEGY_DT_GROW_ONE_LEAF_HEAVIEST))
            test_leaf_index = heaviest_leaf_in(leaves, try_leaves_indexes, Algorithm.MAX_DEPTH_DT);
          else
            Dataset.perror(
                "Discriminator_Tree.class :: no STRATEGY_DT_GROW = "
                    + myBoost.strategy_dt_grow_one_leaf);
        } catch (NoLeafFoundException ee) {
          // only split_pos_pure_leaves found in this case
          return output_vector_grow(Algorithm.NO_DT_SPLIT_NO_SPLITTABLE_LEAF_FOUND, null);
        }

        check_contain = try_leaves_indexes.remove(new Integer(test_leaf_index));
        if (!check_contain)
          Dataset.perror(
              "Discriminator_Tree.class :: should contain " + test_leaf_index + " in leaves");

        // test nn as leaf to split
        nn = ((Discriminator_Node) leaves.elementAt(test_leaf_index));
        deltabest = -1.0;
        split_characteristics_best = null;

        if (!nn.is_pure()) {
          System.out.print("[");
          for (i_split = 0; i_split < myBoost.myDomain.myDS.number_domain_features(); i_split++) {
            f_split = nn.support_at_classification_node.elementAt(i_split);
            if ((!Feature.HAS_SINGLETON_DOMAIN(f_split)) && (Feature.SPLIT_AUTHORIZED(f_split))) {

              if ((RANDOMISE_SPLIT_FINDING_WHEN_TOO_MANY_SPLITS)
                  && (f_split.tests.size() > MAX_SPLITS_BEFORE_RANDOMISATION)) {
                n_tests = MAX_SPLITS_BEFORE_RANDOMISATION;
                Algorithm.FEAT_SIZE++;
                System.out.print("-");
              } else {
                System.out.print("+");

                n_tests = f_split.tests.size();
              }

              for (j_split = 0; j_split < n_tests; j_split++) {
                if ((RANDOMISE_SPLIT_FINDING_WHEN_TOO_MANY_SPLITS)
                    && (f_split.tests.size() > MAX_SPLITS_BEFORE_RANDOMISATION))
                  j_split_index = Algorithm.R.nextInt(f_split.tests.size());
                else j_split_index = j_split;
                split_characteristics =
                    nn.split_characteristics(i_split, j_split_index, alpha_leaf, true);
                if (nn.split_is_eligible(split_characteristics)) {
                  deltacur = ((Double) split_characteristics.elementAt(0)).doubleValue();

                  if ((split_characteristics_best == null) || (deltacur > deltabest)) {
                    deltabest = deltacur;
                    split_characteristics_best =
                        Discriminator_Node.SURROGATE_split_characteristics(split_characteristics);
                  }
                }
              }
            } else if (!Feature.SPLIT_AUTHORIZED(f_split)) System.out.print("X");
            else System.out.print("O");
          }
          System.out.print("]");

          if (split_characteristics_best == null) {
            if (split_leaves_with_no_impure_split == null)
              split_leaves_with_no_impure_split = new Vector<Integer>();
            split_leaves_with_no_impure_split.addElement(new Integer(test_leaf_index));

            Dataset.warning("DT leaf " + nn + " admits no split with both leaves not +pure");
            found_leaf = false;
          } else if (!Discriminator_Tree.SPLIT(split_characteristics_best))
            Dataset.perror("Discriminator_Tree.class :: kept a best split which is not eligible");
        } else {
          Algorithm.FEAT_SIZE++; // records the # pure leaves (inclues real AND generated)

          if (pure_leaves == null) pure_leaves = new Vector<Integer>();
          pure_leaves.addElement(new Integer(test_leaf_index));

          found_leaf = false;
        }
      } while ((!found_leaf) && (try_leaves_indexes.size() > 0));

      if (found_leaf) {

        if (nn == null) Dataset.perror("Discriminator_Tree.class :: null leaf to split");

        if (split_characteristics_best == null)
          Dataset.perror("Discriminator_Tree.class :: no test for leaf to split");

        split_characteristics = split_characteristics_best;

        nnret = nn;

        nn.is_leaf = false;
        nn.link_value = 0.0;
        nn.feature_node_index = ((Integer) split_characteristics.elementAt(2)).intValue();
        nn.feature_node_test_index = ((Integer) split_characteristics.elementAt(3)).intValue();

        pos_left = ((Integer) split_characteristics.elementAt(6)).intValue();
        neg_left = ((Integer) split_characteristics.elementAt(7)).intValue();
        pos_right = ((Integer) split_characteristics.elementAt(8)).intValue();
        neg_right = ((Integer) split_characteristics.elementAt(9)).intValue();

        wpos_left = ((Double) split_characteristics.elementAt(10)).doubleValue();
        wneg_left = ((Double) split_characteristics.elementAt(11)).doubleValue();
        wpos_right = ((Double) split_characteristics.elementAt(12)).doubleValue();
        wneg_right = ((Double) split_characteristics.elementAt(13)).doubleValue();

        split_supports =
            Feature.SPLIT_SUPPORT_DT_INDUCTION(
                nn, nn.feature_node_index, nn.feature_node_test_index, myBoost.myDomain.myDS);
        Feature.TEST_UNION(
            nn.support_at_classification_node.elementAt(nn.feature_node_index),
            (Feature) split_supports[0].elementAt(nn.feature_node_index),
            (Feature) split_supports[1].elementAt(nn.feature_node_index));

        // updates for left child
        number_nodes++;
        left = (Vector) split_characteristics.elementAt(4);
        leftnn =
            new Discriminator_Node(
                this,
                number_nodes,
                nn.depth + 1,
                split_CV,
                (Vector) left.elementAt(0),
                (Vector) left.elementAt(1),
                pos_left,
                neg_left,
                wpos_left,
                wneg_left,
                split_supports[0]);

        leftnn.compute_link(alpha_leaf);

        // updates REAL training examples_leaves associated to the new left leaf
        dummy_v = (Vector) left.elementAt(0);
        for (ll = 0; ll < dummy_v.size(); ll++) {
          e =
              myBoost.myDomain.myDS.train_example(
                  split_CV, ((Integer) dummy_v.elementAt(ll)).intValue(), true);
          e.update_training_leaf(leftnn);
        }

        // updates for right child
        number_nodes++;
        right = (Vector) split_characteristics.elementAt(5);
        rightnn =
            new Discriminator_Node(
                this,
                number_nodes,
                nn.depth + 1,
                split_CV,
                (Vector) right.elementAt(0),
                (Vector) right.elementAt(1),
                pos_right,
                neg_right,
                wpos_right,
                wneg_right,
                split_supports[1]);

        rightnn.compute_link(alpha_leaf);

        // updates REAL training examples_leaves associated to the new right leaf
        dummy_v = (Vector) right.elementAt(0);
        for (ll = 0; ll < dummy_v.size(); ll++) {
          e =
              myBoost.myDomain.myDS.train_example(
                  split_CV, ((Integer) dummy_v.elementAt(ll)).intValue(), true);
          e.update_training_leaf(rightnn);
        }

        if (nn.depth + 1 > depth) depth = nn.depth + 1;

        nn.left_child = leftnn;
        nn.right_child = rightnn;

        if (new_leaves == null) new_leaves = new Vector<Discriminator_Node>();

        new_leaves.addElement(leftnn);
        new_leaves.addElement(rightnn);
      }
      if ((found_leaf) || (try_leaves_indexes.size() == 0)) stop = true;
    } while (!stop);

    if ((new_leaves == null) || (new_leaves.size() == 0))
      return output_vector_grow(Algorithm.NO_DT_SPLIT_NO_SPLITTABLE_LEAF_FOUND, null);

    stay_leaves = new Vector<Discriminator_Node>();

    // adds old leaves not(split) and not(tried but leading to pure leaf)
    for (j = 0; j < try_leaves_indexes.size(); j++)
      stay_leaves.addElement(
          (Discriminator_Node) leaves.elementAt(try_leaves_indexes.elementAt(j).intValue()));

    // adds split_leaves_with_no_impure_split: old leaves without ANY possible split
    if (split_leaves_with_no_impure_split != null)
      for (i = 0; i < split_leaves_with_no_impure_split.size(); i++)
        stay_leaves.addElement(
            (Discriminator_Node)
                leaves.elementAt(split_leaves_with_no_impure_split.elementAt(i).intValue()));

    // adds pure leaves
    if (pure_leaves != null)
      for (i = 0; i < pure_leaves.size(); i++)
        stay_leaves.addElement(
            (Discriminator_Node) leaves.elementAt(pure_leaves.elementAt(i).intValue()));

    // adds new leaves
    if (new_leaves != null)
      for (i = 0; i < new_leaves.size(); i++)
        stay_leaves.addElement((Discriminator_Node) new_leaves.elementAt(i));

    leaves = stay_leaves;
    compute_leaves_hashtable();

    new_leaves = null;
    pure_leaves = null;
    split_leaves_with_no_impure_split = null;
    stay_leaves = null;
    left = right = null;
    dummy_v = null;
    split_supports = null;
    try_leaves_indexes = null;
    split_characteristics = split_characteristics_best = null;

    return output_vector_grow(Algorithm.DT_SPLIT_OK, nnret);
  }

  public Vector one_step_grow() {
    // the returned vectors =
    // 1 String characterizing the attempt to grow
    // 1 Discriminator_Node, the leaf that has been split (if it happened)
    return one_step_grow_cheaper();
  }

  public void compute_train_fold_indexes_for_fakes_in_nodes_from(
      Discriminator_Node start, int fold) {
    int[] all_indexes_fake, all_indexes_left_fake, all_indexes_right_fake;
    Example e;
    Feature f;
    int i;
    int n_fake = 0;
    double w_fake = 0.0;
    Vector<Integer> go_left_fake = null, go_right_fake = null;

    Discriminator_Node the_one_to_be_modified;
    if (start == null) {
      the_one_to_be_modified = root;
      if (fold != root.split_CV)
        Dataset.perror("Discriminator_Tree.modifying the root on a different CV fold");
      all_indexes_fake = new int[myBoost.myDomain.myDS.train_size(-1, false)];
      for (i = 0; i < myBoost.myDomain.myDS.train_size(-1, false); i++) all_indexes_fake[i] = i;
    } else {
      the_one_to_be_modified = start;
      all_indexes_fake = start.train_fold_indexes_in_node_fake;
    }

    for (i = 0; i < all_indexes_fake.length; i++) {
      e = myBoost.myDomain.myDS.train_example(-1, all_indexes_fake[i], false);
      if (e.is_positive())
        Dataset.perror("Discriminator_Tree.class :: only for generated examples");
      else {
        n_fake++;
        w_fake += e.unnormalized_weight;
      }
      if (!the_one_to_be_modified.is_leaf) {
        f =
            the_one_to_be_modified.support_at_classification_node.elementAt(
                the_one_to_be_modified.feature_node_index);

        // check that no uncertainty in split
        if (Example.FEATURE_IS_UNKNOWN(e, the_one_to_be_modified.feature_node_index))
          Dataset.perror(
              "Discriminator_Tree.class :: generated example cannot have unkown feature value");

        if (f.example_goes_left(
            e,
            the_one_to_be_modified.feature_node_index,
            the_one_to_be_modified.feature_node_test_index,
            true)) {
          if (go_left_fake == null) go_left_fake = new Vector<>();
          go_left_fake.addElement(new Integer(all_indexes_fake[i]));
        } else {
          if (go_right_fake == null) go_right_fake = new Vector<>();
          go_right_fake.addElement(new Integer(all_indexes_fake[i]));
        }
      }
    }

    the_one_to_be_modified.n_fake = n_fake;
    the_one_to_be_modified.w_fake = w_fake;

    the_one_to_be_modified.compute_link(myBoost.getAlpha(this));

    if (!the_one_to_be_modified.is_leaf) {
      if ((go_left_fake != null) && (the_one_to_be_modified.left_child == null))
        Dataset.perror("Discriminator_Tree.class :: no left child but examples going left");

      if ((go_right_fake != null) && (the_one_to_be_modified.right_child == null))
        Dataset.perror("Discriminator_Tree.class :: no right child but examples going right");

      if (go_left_fake != null) {
        all_indexes_left_fake = new int[go_left_fake.size()];
        for (i = 0; i < go_left_fake.size(); i++)
          all_indexes_left_fake[i] = go_left_fake.elementAt(i).intValue();
        the_one_to_be_modified.left_child.train_fold_indexes_in_node_fake = all_indexes_left_fake;
      }

      if (go_right_fake != null) {
        all_indexes_right_fake = new int[go_right_fake.size()];
        for (i = 0; i < go_right_fake.size(); i++)
          all_indexes_right_fake[i] = go_right_fake.elementAt(i).intValue();
        the_one_to_be_modified.right_child.train_fold_indexes_in_node_fake = all_indexes_right_fake;
      }

      if (go_left_fake != null)
        compute_train_fold_indexes_for_fakes_in_nodes_from(the_one_to_be_modified.left_child, fold);

      if (go_right_fake != null)
        compute_train_fold_indexes_for_fakes_in_nodes_from(
            the_one_to_be_modified.right_child, fold);
    }
  }

  public void init() {
    int i,
        ne_real = myBoost.myDomain.myDS.train_size(split_CV, true),
        ne_fake = myBoost.myDomain.myDS.train_size(split_CV, false); // total_train_size(split_CV);
    Vector<Integer> indexes_real = new Vector<>();
    Vector<Integer> indexes_fake = new Vector<>();
    Example e;
    int n_real = 0, n_fake = 0;
    double w_real = 0.0, w_fake = 0.0, alpha_leaf;
    Vector<Feature> sn = new Vector<>();
    Feature f_node;

    for (i = 0; i < ne_real; i++) {
      indexes_real.addElement(new Integer(i));
      e = myBoost.myDomain.myDS.train_example(split_CV, i, true);
      n_real++;
      w_real += e.unnormalized_weight;
    }

    for (i = 0; i < ne_fake; i++) {
      indexes_fake.addElement(new Integer(i));
      e = myBoost.myDomain.myDS.train_example(split_CV, i, false);
      n_fake++;
      w_fake += e.unnormalized_weight;
    }

    unnormalized_weights_sum = w_real + w_fake;

    number_nodes = 1;

    for (i = 0; i < myBoost.myDomain.myDS.number_domain_features(); i++) {
      f_node = Feature.copyOf((Feature) myBoost.myDomain.myDS.domain_feature(i), true);
      f_node.try_format_tests(myBoost.myDomain.myDS, i, true);

      sn.addElement(f_node);
    }

    root =
        new Discriminator_Node(
            this,
            number_nodes,
            0,
            split_CV,
            indexes_real,
            indexes_fake,
            n_real,
            n_fake,
            w_real,
            w_fake,
            sn);
    init_real_examples_training_leaf(root);

    depth = 0;

    alpha_leaf = myBoost.getAlpha(this);

    root.compute_link(alpha_leaf);

    leaves = new Vector<>();
    leaves.addElement(root);
    compute_leaves_hashtable();
  }

  public Discriminator_Node get_leaf(Example ee) {
    // returns the leaf reached by the example
    Discriminator_Node nn = root;
    Feature f;
    while (!nn.is_leaf) {
      f = nn.support_at_classification_node.elementAt(nn.feature_node_index);

      if (f.example_goes_left(ee, nn.feature_node_index, nn.feature_node_test_index, true))
        nn = nn.left_child;
      else nn = nn.right_child;
    }
    return nn;
  }
}
