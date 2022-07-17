// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Generator_Tree
 *****/

class Generator_Tree implements Debuggable {
  public static boolean IMPUTATION_AT_MAXIMUM_LIKELIHOOD = true;

  public static int DISPLAY_TREE_TYPE = 1;
  // 1: does not display leaves support, just list leaves

  int name, depth;

  Generator_Node root;
  // root of the tree

  HashSet<Generator_Node> leaves;
  // list of leaves of the tree (potential growth here)

  Boost myBoost;

  int max_depth;

  int number_nodes;
  // includes leaves

  double entropy;
  // entropy at the leaves

  Vector<Histogram> gt_histograms;

  public static void SAVE_GENERATOR_TREE(Generator_Tree gt, String nameSave, String saveString) {
    // saves a generator in a proper format
    int i;
    FileWriter f = null;
    try {
      f = new FileWriter(nameSave);
      f.write(saveString);

      f.write("\n// Tree description (do not change)\n\n");

      if (gt.root == null) {
        f.write(Dataset.KEY_NODES + "\n");
        f.write("null\n");
      } else {
        // saving tree architecture
        f.write(Dataset.KEY_NODES + "\n");
        gt.root.recursive_write_nodes(f);

        // saving arcs features
        f.write("\n" + Dataset.KEY_ARCS + "\n");
        gt.root.recursive_write_arcs(f);
      }
      f.close();
    } catch (IOException e) {
      Dataset.perror("Generator_Tree.class :: Saving results error in file " + nameSave);
    }
  }

  Generator_Tree(int nn, Boost bb, int maxs) {
    name = nn;
    root = null;
    myBoost = bb;

    max_depth = maxs;
    depth = -1; // Generator empty
    gt_histograms = null;
  }

  public void compute_generator_histograms() {
    gt_histograms = new Vector<>();
    Vector<Example> ex_for_histogram =
        generate_sample(Dataset.NUMBER_GENERATED_EXAMPLES_DEFAULT_HISTOGRAMS);

    Histogram h;
    int i;
    boolean ok;
    for (i = 0; i < myBoost.myDomain.myDS.number_domain_features(); i++) {
      h = myBoost.myDomain.myDS.from_histogram(i);
      ok = h.fill_histogram(ex_for_histogram, true);
      if (!ok)
        Dataset.perror(
            "Dataset.class :: not a single domain example with non UNKNOWN feature value for"
                + " feature #"
                + i);
      gt_histograms.addElement(h);
    }
  }

  public Generator_Tree surrogate() {
    boolean tr;

    Generator_Tree gt = new Generator_Tree(-1, myBoost, max_depth);
    gt.init();

    if (leaves.size() > 1) {
      tr = gt.leaves.remove(gt.root);
      if (!tr) Dataset.perror("Generator_Tree.class :: only one leaf but root not inside");
    }

    gt.root.recursive_copy(root, null);

    gt.depth = depth;
    gt.number_nodes = number_nodes;
    gt.entropy = entropy;

    int i;
    gt.gt_histograms = new Vector<>();
    for (i = 0; i < myBoost.myDomain.myDS.number_domain_features(); i++)
      gt.gt_histograms.addElement(Histogram.copyOf(gt_histograms.elementAt(i)));

    return gt;
  }

  public Generator_Node find_leaf(Generator_Node model_leaf) {
    // find in this the *leaf* whose name matches model.name

    Generator_Node v = null, cur;
    Iterator<Generator_Node> it = leaves.iterator();
    while (it.hasNext()) {
      cur = it.next();
      if ((cur.name == model_leaf.name) && (v != null))
        Dataset.perror("Generator_Tree.class :: find_leaf found >1 matching leaves");
      else if (cur.name == model_leaf.name) v = cur;
    }
    return v;
  }

  public Vector<Generator_Node> all_leaves_that_could_generate(Example e) {
    Vector<Generator_Node> ret = new Vector<Generator_Node>();
    Generator_Node gn;

    Iterator<Generator_Node> it = leaves.iterator();
    while (it.hasNext()) {
      gn = it.next();
      if (Generator_Node.LEAF_COULD_GENERATE_EXAMPLE(gn, e)) ret.addElement(gn);
    }

    if (ret.size() == 0) Dataset.perror("Generator_Tree.class :: no leaf to generate example " + e);
    return ret;
  }

  public void impute_all_values_from_one_leaf(Example e) {
    Vector<Generator_Node> dumgn = all_leaves_that_could_generate(e);
    double[] all_ps = new double[dumgn.size()];
    int i, j, index, nmax, index_chosen, count;
    double tot_weight = 0.0, dindex;

    for (i = 0; i < dumgn.size(); i++) {

      all_ps[i] = dumgn.elementAt(i).p_node;
      tot_weight += all_ps[i];
    }

    if (Statistics.APPROXIMATELY_EQUAL(tot_weight, 0.0, EPS4))
      Dataset.perror(
          "Generator_Tree.class :: sum of generating nodes p_node = "
              + tot_weight
              + "  approx 0.0");

    for (i = 0; i < dumgn.size(); i++) all_ps[i] /= tot_weight;

    double cp = all_ps[0];
    double p = Algorithm.R.nextDouble();
    boolean[] is_max = new boolean[dumgn.size()];

    if (Generator_Tree.IMPUTATION_AT_MAXIMUM_LIKELIHOOD) {
      for (i = 0; i < dumgn.size(); i++) all_ps[i] /= dumgn.elementAt(i).node_volume();
      index = 0;
      for (i = 0; i < dumgn.size(); i++) if (all_ps[i] >= all_ps[index]) index = i;

      // picks a leaf at random among *all* maxes
      nmax = 0;
      count = -1;
      for (i = 0; i < dumgn.size(); i++)
        if (all_ps[i] == all_ps[index]) {
          is_max[i] = true;
          nmax++;
        } else is_max[i] = false;

      index_chosen = Math.abs(Algorithm.R.nextInt()) % nmax;

      i = 0;
      do {
        if (all_ps[i] == all_ps[index]) count++;
        if (count < index_chosen) i++;
      } while (count < index_chosen);

      if (all_ps[i] != all_ps[index])
        Dataset.perror("Genrator_Tree.class :: not a max chosen in index " + i);

    } else {
      i = 0;
      while (p > cp) {
        i++;
        cp += all_ps[i];
      }
    }
    Feature f;

    if (dumgn.elementAt(i).all_features_domain == null)
      Dataset.perror(
          "Generator_Node.class :: GT leaf "
              + dumgn.elementAt(i).name
              + " has all_features_domain == null");

    Vector<Feature> v = dumgn.elementAt(i).all_features_domain;
    // contains all domains used to impute e;

    for (j = 0; j < e.typed_features.size(); j++)
      if (Unknown_Feature_Value.IS_UNKNOWN(e.typed_features.elementAt(j))) {
        f = v.elementAt(j);

        if (Feature.IS_NOMINAL(f.type)) {
          index = (Math.abs(Algorithm.R.nextInt())) % f.modalities.size();
          e.typed_features.setElementAt(new String(f.modalities.elementAt(index)), j);
        } else if (Feature.IS_INTEGER(f.type)) {
          index = (Math.abs(Algorithm.R.nextInt())) % (f.imax - f.imin + 1);
          e.typed_features.setElementAt(new Integer(f.imin + index), j);
        } else if (Feature.IS_CONTINUOUS(f.type)) {
          dindex = (Algorithm.R.nextDouble()) * (f.dmax - f.dmin);
          e.typed_features.setElementAt(new Double(f.dmin + dindex), j);
        } else Dataset.perror("Generator_Tree.class :: Feature " + f + "'s type not known");
      }
  }

  public Object impute_value(Example e, int feature_index) {
    // returns a possible imputed value given the gt for feature feature_index in e

    if (!Example.FEATURE_IS_UNKNOWN(e, feature_index))
      Dataset.perror(
          "Generator_Tree.class :: Example ("
              + e
              + ")["
              + feature_index
              + "] != "
              + Unknown_Feature_Value.S_UNKNOWN);

    Vector<Generator_Node> dumgn = all_leaves_that_could_generate(e);
    double[] all_ps = new double[dumgn.size()];
    int i, j, index;
    double tot_weight = 0.0, dindex;

    for (i = 0; i < dumgn.size(); i++) {
      all_ps[i] = dumgn.elementAt(i).p_node;
      tot_weight += all_ps[i];
    }

    if (Statistics.APPROXIMATELY_EQUAL(tot_weight, 0.0, EPS4))
      Dataset.perror(
          "Generator_Tree.class :: sum of generating nodes p_node = "
              + tot_weight
              + "  approx 0.0");

    for (i = 0; i < dumgn.size(); i++) all_ps[i] /= tot_weight;

    double cp = all_ps[0];
    double p = Algorithm.R.nextDouble();
    i = 0;
    while (p > cp) {
      i++;
      cp += all_ps[i];
    }

    if (dumgn.elementAt(i).all_features_domain == null)
      Dataset.perror(
          "Generator_Node.class :: GT leaf "
              + dumgn.elementAt(i).name
              + " has all_features_domain == null");

    Vector<Feature> v = dumgn.elementAt(i).all_features_domain;

    Feature f = v.elementAt(feature_index);

    if (Feature.IS_NOMINAL(f.type)) {
      index = (Math.abs(Algorithm.R.nextInt())) % f.modalities.size();
      return new String(f.modalities.elementAt(index));
    } else if (Feature.IS_INTEGER(f.type)) {
      index = (Math.abs(Algorithm.R.nextInt())) % (f.imax - f.imin + 1);
      return new Integer(f.imin + index);
    } else if (Feature.IS_CONTINUOUS(f.type)) {
      dindex = (Algorithm.R.nextDouble()) * (f.dmax - f.dmin);
      return new Double(f.dmin + dindex);
    } else Dataset.perror("Generator_Tree.class :: Feature " + f + "'s type not known");

    return null;
  }

  public double[] probability_vector(Example e, Histogram domain_histo, int feature_index) {
    // returns P[F|e,h] where F = feature at feature_index, h = domain_histo provides a binning for
    // continuous / integers

    if (!myBoost.myDomain.myDS.domain_feature(feature_index).name.equals(domain_histo.name))
      Dataset.perror(
          "Generator_Tree.class :: name mismatch to compute probability_vector ("
              + myBoost.myDomain.myDS.domain_feature(feature_index).name
              + " != "
              + domain_histo.name
              + ")");
    int i, j;

    double[] ret = new double[domain_histo.histogram_features.size()];
    double[] fprop;
    double[] all_ps;

    double tot_weight = 0.0;

    Vector<Generator_Node> dumgn = all_leaves_that_could_generate(e);
    all_ps = new double[dumgn.size()];

    for (i = 0; i < dumgn.size(); i++) {
      all_ps[i] = dumgn.elementAt(i).p_node;
      tot_weight += all_ps[i];
    }

    if (Statistics.APPROXIMATELY_EQUAL(tot_weight, 0.0, EPS4))
      Dataset.perror(
          "Generator_Tree.class :: sum of generating nodes p_node = "
              + tot_weight
              + "  approx 0.0");

    for (i = 0; i < dumgn.size(); i++) all_ps[i] /= tot_weight;

    Vector<Feature> dumf = new Vector<>();
    for (i = 0; i < dumgn.size(); i++)
      dumf.addElement(
          Generator_Node.ALL_FEATURES_DOMAINS_AT_SAMPLING_NODE(this, dumgn.elementAt(i))
              .elementAt(feature_index));

    for (i = 0; i < dumgn.size(); i++) {
      fprop = domain_histo.normalized_domain_intersection_with(dumf.elementAt(i));
      for (j = 0; j < fprop.length; j++) ret[j] += (all_ps[i] * fprop[j]);
    }

    return ret;
  }

  public void update_generator_tree(
      Generator_Node leaf, Generator_Node[] created_leaves, Discriminator_Tree dt) {
    // used e.g. in split_leaf to update the GT after a split has been found at leaf leaf and the
    // new leaves are in created_leaves
    // multi_p must have been computed

    if ((leaf.multi_p == null)
        || (!Statistics.APPROXIMATELY_EQUAL(leaf.multi_p[0] + leaf.multi_p[1], 1.0, EPS2)))
      Dataset.perror(
          "Generator_Tree.class :: leaf to be split has multi_p not yet computed or badly"
              + " computed");

    boolean tr;
    int i, base_depth = -1;

    // update leaves set + p_nodes
    leaf.is_leaf = false;
    leaf.all_features_domain = null;

    tr = leaves.remove(leaf);
    if (!tr) Dataset.perror("Generator_Tree.class :: leaf not found");
    for (i = 0; i < created_leaves.length; i++) {
      // check depth
      if (i == 0) base_depth = created_leaves[i].depth;
      else if (base_depth != created_leaves[i].depth)
        Dataset.perror("Generator_Tree.class :: non-consistant depth in new nodes");

      tr = leaves.add(created_leaves[i]);
      if (!tr) Dataset.perror("Generator_Tree.class :: duplicated leaf found");
      created_leaves[i].p_node = leaf.p_node * leaf.multi_p[i];
      created_leaves[i].compute_all_features_domain();
    }

    // update depth, number_nodes
    if (base_depth == -1) Dataset.perror("Generator_Tree.class :: no base depth computed");

    if (depth < base_depth) depth = base_depth;
    number_nodes += created_leaves.length;
  }

  public Generator_Node get_leaf_to_be_split(
      Discriminator_Tree dt, Discriminator_Node node_just_split) {
    // WARNING: make sure names of nodes are 1-1 between DT and GT
    // find the corresponding leaf to split in GT and returns it

    if (node_just_split.is_leaf)
      Dataset.perror(
          "Generator_Tree.class :: copycat DT node is a leaf, while it should be a stump");

    Iterator<Generator_Node> it = leaves.iterator();
    Generator_Node target_gt = null, candidate_gt;
    boolean found = false;
    while ((it.hasNext()) && (!found)) {
      candidate_gt = it.next();
      if (candidate_gt.name == node_just_split.name) {
        target_gt = candidate_gt;
        found = true;
      }
    }

    if (!found)
      Dataset.perror(
          "Generator_Tree.class :: copycat does not find node #"
              + node_just_split.name
              + " in the leaves of "
              + this);
    return target_gt;
  }

  public String one_step_grow_copycat(
      Discriminator_Tree dt, Discriminator_Node node_just_split, Generator_Node leaf_to_be_split) {
    // WARNING: make sure names of nodes are 1-1 between DT and GT
    // find the corresponding leaf to split in GT

    if (node_just_split.is_leaf)
      Dataset.perror(
          "Generator_Tree.class :: copycat DT node is a leaf, while it should be a stump");

    int findex;
    Vector<Feature> cv;
    Feature gt_leaf_feature;
    Generator_Node[] created_leaves;
    double p_final;

    Generator_Node target_gt = leaf_to_be_split;

    // makes sure that both leaves attached to node_just_split have positive examples reaching them
    if ((!Boost.AUTHORISE_REAL_PURE_LEAVES)
        && ((node_just_split.left_child.w_real == 0.0)
            || (node_just_split.right_child.w_real == 0.0)))
      Dataset.perror(
          "Generator_Tree.class :: copycat stump "
              + node_just_split
              + " in tree "
              + dt
              + " has at least one real-pure leaf");

    // test features at the nodes of the stump at node_just_split
    if ((node_just_split.support_at_classification_node.size()
            != myBoost.myDomain.myDS.number_domain_features())
        || (node_just_split.left_child.support_at_classification_node.size()
            != myBoost.myDomain.myDS.number_domain_features())
        || (node_just_split.right_child.support_at_classification_node.size()
            != myBoost.myDomain.myDS.number_domain_features()))
      Dataset.perror(
          "Generator_Tree.class :: copycat stump has at non-valid feature domain vector sizes");

    findex = node_just_split.feature_node_index;

    gt_leaf_feature = Generator_Node.FEATURE_DOMAIN_AT_SAMPLING_NODE(this, target_gt, findex);

    cv = new Vector<>();
    cv.addElement(
        (Feature) node_just_split.left_child.support_at_classification_node.elementAt(findex));
    cv.addElement(
        (Feature) node_just_split.right_child.support_at_classification_node.elementAt(findex));

    p_final =
        node_just_split.right_child.w_real
            / (node_just_split.left_child.w_real + node_just_split.right_child.w_real);

    created_leaves = target_gt.split(findex, gt_leaf_feature, p_final, cv);
    // changing names eventually for the new leaves to match the DT's names
    created_leaves[0].name = node_just_split.left_child.name;
    created_leaves[1].name = node_just_split.right_child.name;
    root.check_names(new HashSet<Integer>());

    update_generator_tree(target_gt, created_leaves, dt);

    gt_leaf_feature = null;
    cv = null;
    created_leaves = null;

    return Algorithm.GT_SPLIT_OK;
  }

  public void check_recursive(Generator_Node gn) {
    int i;
    if (!gn.is_leaf) {
      if ((gn.children_arcs == null) || (gn.children_arcs.length < 2))
        Dataset.perror("Generator_Tree.class :: " + gn + " not a leaf but without children");
      for (i = 0; i < gn.children_arcs.length; i++) check_recursive(gn.children_arcs[i].end_node);
    } else {
      if (!leaves.contains(gn))
        Dataset.perror("Generator_Tree.class :: " + gn + " a leaf but not in *leaves*");
    }
  }

  public void init() {
    number_nodes = 1;
    boolean tr;

    root = new Generator_Node(this, null, number_nodes, 0, -1);
    root.compute_all_features_domain();

    depth = 0;

    leaves = new HashSet<>();
    tr = leaves.add(root);
    if (!tr) Dataset.perror("Generator_Tree.class :: adding root failed");
  }

  public boolean has_root(Generator_Node gn) {
    // simple implementation
    if ((gn.myParentGenerator_Node_children_number == -1) && (gn.name != 1))
      Dataset.perror("Generator_Tree.class :: bad root IDs");

    if (gn.myParentGenerator_Node_children_number == -1) return true;
    return false;
  }

  public String toString() {
    boolean leaf_as_well = true, compute_support_for_leaf = true;

    if (Generator_Tree.DISPLAY_TREE_TYPE == 1) leaf_as_well = compute_support_for_leaf = false;

    int i, j;
    String v = "(name = #" + name + " | depth = " + depth + " | #nodes = " + number_nodes + ")";
    boolean stop = false;

    Generator_Node n = root;
    Vector<Generator_Node> cur_nodes = new Vector<>();
    cur_nodes.addElement(n);

    Vector<Generator_Node> next_nodes;
    Iterator it;

    v += root.display(new HashSet());

    if (!leaf_as_well) {
      v += "\nSampling Leaves: ";
      it = leaves.iterator();
      while (it.hasNext()) {
        n = (Generator_Node) it.next();
        v += "(#" + n.name + ":" + n.depth + ":" + DF4.format(n.p_node) + ") ";
      }
      v += "\n";
    }

    return v;
  }

  public Generator_Node sample_leaf() {
    return sample_leaf(root);
  }

  public Generator_Node sample_leaf(Generator_Node cn) {
    // same as sample_leaf but starts from cn;
    double p, cp;
    int i;
    while (!cn.is_leaf) {
      cp = cn.multi_p[0];
      p = Algorithm.R.nextDouble();
      i = 0;
      while (p > cp) {
        i++;
        cp += cn.multi_p[i];
      }
      if (i >= cn.children_arcs.length)
        Dataset.perror(
            "Generator_Tree.class :: children index " + i + " >= max = " + cn.children_arcs.length);

      cn = cn.children_arcs[i].end_node;
    }
    return cn;
  }

  public Vector sample_leaf_with_density() {
    double p, cp, densval = 1.0;
    Generator_Node cn = root;
    Vector ret = new Vector();

    int i;
    while (!cn.is_leaf) {
      cp = cn.multi_p[0];
      p = Algorithm.R.nextDouble();
      i = 0;
      while (p > cp) {
        i++;
        cp += cn.multi_p[i];
      }
      if (i >= cn.children_arcs.length)
        Dataset.perror(
            "Generator_Tree.class :: children index " + i + " >= max = " + cn.children_arcs.length);
      densval *= cn.multi_p[i];
      cn = cn.children_arcs[i].end_node;
    }

    ret.addElement(cn);
    ret.addElement(new Double(densval));

    return ret;
  }

  public Vector<Example> generate_sample(int nex) {
    int i;
    Vector<Example> set = new Vector<>();
    Generator_Node sample_leaf;
    Example e;
    for (i = 0; i < nex; i++) {
      sample_leaf = sample_leaf();
      e = sample_leaf.one_example(i);
      if ((!SAVE_TIME)
          || (Math.abs(Algorithm.R.nextInt()) % 1000
              == 0)) { // makes a random "quality" check on generated examples
        if (!Generator_Node.LEAF_COULD_GENERATE_EXAMPLE(sample_leaf, e))
          Dataset.perror("Generator_Tree.class :: leaf " + sample_leaf + " cannot generate " + e);
      }
      set.addElement(e);
    }

    return set;
  }

  public void replace_sample(Vector<Example> generated_examples, Generator_Node gn) {
    if (gn.is_leaf) Dataset.perror("Generator_Tree.class :: node " + gn + " should not be a leaf");

    int i, findex;
    Vector<Example> set = new Vector<>();
    Generator_Node sample_leaf;
    Example e;
    for (i = 0; i < generated_examples.size(); i++) {
      e = generated_examples.elementAt(i);
      if (e.generating_leaf.name == gn.name) {

        // We replace the feature of the example now at the split in gn
        findex = gn.feature_node_index;
        sample_leaf = sample_leaf(gn);
        sample_leaf.resample_feature_in_generated_example(e, findex);
      }
    }
  }

  public Vector<Example> generate_sample_with_density(int nex) {
    int i;
    Vector<Example> set = new Vector<>();
    Vector bundle;
    Generator_Node sample_leaf;
    Example e;
    for (i = 0; i < nex; i++) {
      bundle = sample_leaf_with_density();

      sample_leaf = (Generator_Node) bundle.elementAt(0);
      e = sample_leaf.one_example(i);
      e.local_density = ((Double) bundle.elementAt(1)).doubleValue();

      if (!Generator_Node.LEAF_COULD_GENERATE_EXAMPLE(sample_leaf, e))
        Dataset.perror("Generator_Tree.class :: leaf " + sample_leaf + " cannot generate " + e);
      set.addElement(e);
    }
    return set;
  }
}
