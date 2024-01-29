import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class MonotonicTreeGraph ( = Monotonic Decision Tree) and related classes
 *****/

class HyperbolicPoint2D implements Debuggable {
  // done so to be able to embed BigDecimal instead of doubles

  public static double DEFAULT_ANGLE_ROOT_TO_X_AXIS = -Math.PI / 2.0;

  public static double SARKAR_EPSILON =
      0.1; // precision for farther nodes, see "Representation Tradeoffs for Hyperbolic Embeddings"

  double x, y;

  public static HyperbolicPoint2D POINCARE_NORMALIZATION(HyperbolicPoint2D z) {
    double factor = 1.0 / z.euclid_norm_squared();

    return z.times(factor);
  }

  public static double FACTOR_INVERSION(HyperbolicPoint2D center, HyperbolicPoint2D x) {
    if ((center.x == x.x) && (center.y == x.y))
      Dataset.perror("HyperbolicPoint2D.class :: identical coordinates -- no inversion factor");

    double num = center.euclid_norm_squared() - 1.0;
    double den = center.euclid_distance_squared(x);

    return num / den;
  }

  public static HyperbolicPoint2D POINCARE_ROOT(MonotonicTreeNode node) {
    double quality = Math.abs(node.alpha_value);
    double r = (Math.exp(quality) - 1.0) / (Math.exp(quality) + 1.0);

    return new HyperbolicPoint2D(
        r * Math.cos(DEFAULT_ANGLE_ROOT_TO_X_AXIS), r * Math.sin(DEFAULT_ANGLE_ROOT_TO_X_AXIS));
  }

  public static HyperbolicPoint2D SARKAR_MAP_GRANDCHILDREN(double distance, double angle) {
    double r = Math.tanh(distance);

    return new HyperbolicPoint2D(r * Math.cos(angle), r * Math.sin(angle));
  }

  HyperbolicPoint2D() {
    x = y = 0.0;
  }

  HyperbolicPoint2D(double x, double y) {
    this.x = x;
    this.y = y;
  }

  public String toString() {
    return "[" + DF8.format(x) + ", " + DF8.format(y) + "]";
  }

  public double euclid_norm() {
    return Math.sqrt((x * x) + (y * y));
  }

  public double dot(HyperbolicPoint2D z) {
    return ((x * z.x) + (y * z.y));
  }

  public double euclid_norm_squared() {
    return (euclid_norm() * euclid_norm());
  }

  public double euclid_distance(HyperbolicPoint2D a) {
    return subtract(a).euclid_norm();
  }

  public double angle(HyperbolicPoint2D ref) {
    HyperbolicPoint2D translated = subtract(ref);

    double theta = Math.acos(translated.x / translated.euclid_norm());

    if (translated.y < 0.0) theta = (2.0 * Math.PI) - theta;
    return theta;
  }

  public static double ACOSH(double d) {
    if (Math.abs(d) < 1.0) Dataset.perror("HyperbolicPoint2D.class :: acosh(" + d + ") undefined");

    return Math.log(d + Math.sqrt((d * d) - 1.0));
  }

  public double Poincare_hyperbolic_distance(HyperbolicPoint2D a) {
    double sqd = euclid_distance_squared(a);
    double rat = 2.0 * sqd / ((1.0 - euclid_norm_squared()) * (1.0 - a.euclid_norm_squared()));

    return HyperbolicPoint2D.ACOSH(1.0 + rat);
  }

  public double euclid_distance_squared(HyperbolicPoint2D a) {
    return (euclid_distance(a) * euclid_distance(a));
  }

  public double parent_angle_after_inversion(HyperbolicPoint2D child) {
    HyperbolicPoint2D p_after =
        PoincareDiskEmbeddingUI.POINCARE_CIRCLE_INVERSION_ORIGIN_MODELE(
            child, this); // REMOVE: USE angle instead

    double v = Math.acos(p_after.x / p_after.euclid_norm());
    if (p_after.y < 0) v = (2.0 * Math.PI) - v;
    return v;
  }

  public HyperbolicPoint2D times(double a) {
    return new HyperbolicPoint2D(a * x, a * y);
  }

  public HyperbolicPoint2D add(HyperbolicPoint2D a) {
    return new HyperbolicPoint2D(x + a.x, y + a.y);
  }

  public HyperbolicPoint2D subtract(HyperbolicPoint2D a) {
    return new HyperbolicPoint2D(x - a.x, y - a.y);
  }
}

class MonotonicTreeNode implements Debuggable {

  DecisionTreeNode handle;
  // gives access to all the information needed

  double alpha_value;
  HyperbolicPoint2D embedding_coordinates;
  double fan_for_plotting;

  Vector<DecisionTreeSkipTreeArc> children_arcs;
  HashSet<DecisionTreeNode> leaves_blobbed;

  boolean is_leaf, is_root;
  // is_root used for display purposes, to optimize Sarkar's construction

  int depth, degree;

  public static MonotonicTreeNode FAKE_NODE() {
    return new MonotonicTreeNode();
  }

  MonotonicTreeNode() {
    // creates a fake node, to be used only for display purposes
    handle = null;
    alpha_value = 0.0;
    embedding_coordinates = null;
    children_arcs = null;
    leaves_blobbed = null;
    is_leaf = false;
    depth = degree = -1;
    is_root = false;
    fan_for_plotting = -1.0;
  }

  MonotonicTreeNode(String which_alpha, DecisionTreeNode h) {
    handle = h;

    if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_CARDINALS))
      alpha_value = h.node_prediction_from_cardinals;
    else if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_BOOSTING_WEIGHTS))
      alpha_value = h.node_prediction_from_boosting_weights;
    else Dataset.perror("MonotonicTreeGraph.class :: no such prediction as " + which_alpha);

    embedding_coordinates = null;
    children_arcs = null;
    leaves_blobbed = null;
    is_leaf = false;
    depth = degree = -1;
    is_root = false;
  }

  MonotonicTreeNode(String which_alpha, DecisionTreeNode h, int d) {
    this(which_alpha, h);
    depth = d;
    degree = 0;
  }

  public void count_subleaves(int[] current_total) {
    int i;
    if ((is_leaf) || (children_arcs == null) || (children_arcs.size() == 0)) current_total[0] = 0;
    else
      for (i = 0; i < children_arcs.size(); i++) {
        if (children_arcs.elementAt(i).monotonic_end.is_leaf) current_total[0]++;
        else children_arcs.elementAt(i).monotonic_end.count_subleaves(current_total);
      }
  }

  public double embedded_radius_from_distance(MonotonicTreeNode child, MonotonicTreeGraph mtg) {
    if (Math.abs(child.alpha_value) - Math.abs(alpha_value) < 0.0)
      Dataset.perror(
          "MonotonicTreeNode.class :: my alpha_value ("
              + alpha_value
              + ") is not smaller than child ("
              + child.alpha_value
              + ")");

    double dist = Math.abs(child.alpha_value) - Math.abs(alpha_value);
    return Math.tanh(dist * mtg.distance_scaling() / 2.0);
  }

  public double distance_from_embedded_points(MonotonicTreeNode child, MonotonicTreeGraph mtg) {
    return (embedding_coordinates.Poincare_hyperbolic_distance(child.embedding_coordinates)
        / mtg.distance_scaling());
  }

  public void is_leaf() {
    is_leaf = true;
    if ((embedding_coordinates != null) || (children_arcs != null) || (leaves_blobbed != null))
      Dataset.perror(
          "MonotonicTreeNode.class :: creating a leaf with non-leaf variables instanciated (DT node"
              + " "
              + handle
              + ")");
  }

  public boolean equals(Object o) {
    if (o == this) return true;
    if (!(o instanceof DecisionTreeNode)) return false;
    MonotonicTreeNode ft = (MonotonicTreeNode) o;

    if (((handle != null) && (ft.handle == null)) || ((handle == null) && (ft.handle != null)))
      return false;

    if ((handle == null) && (ft.handle == null)) {
      if ((embedding_coordinates != null)
          || (ft.embedding_coordinates != null)
          || (children_arcs != null)
          || (ft.children_arcs != null)
          || (leaves_blobbed != null)
          || (ft.leaves_blobbed != null))
        Dataset.perror(
            "MonotonicTreeNode.class :: inconsistencies for the comparison of "
                + this
                + " and "
                + ft);
      return true;
    }

    if (handle.equals(ft.handle)) return true;

    return false;
  }

  public String display(
      HashSet<Integer> indexes, String path_from_start_to_end, boolean show_embeddings) {
    String v = "", t;
    int i, j, subleaves = 0;
    HashSet<Integer> dum;
    boolean bdum;

    t = "\u2501";

    for (i = 0; i < depth; i++) {
      if ((i == depth - 1) && (indexes.contains(new Integer(i)))) v += "\u2523" + t;
      else if (i == depth - 1) v += "\u2517" + t;
      else if (indexes.contains(new Integer(i))) v += "\u2503 ";
      else v += "  ";
    }

    int[] test_count = new int[1];
    count_subleaves(test_count); // REMOVE

    v +=
        path_from_start_to_end
            + " "
            + toString()
            + " -- "
            + ((show_embeddings) ? embedding_coordinates : "")
            + ") {{"
            + test_count[0]
            + "}}\n";

    if (!is_leaf) {
      dum = new HashSet<Integer>(indexes);
      bdum = dum.add(new Integer(depth));

      if ((children_arcs == null) || (children_arcs.size() == 0))
        Dataset.perror(
            "MonotonicTreeNode.class :: node pointing at "
                + handle
                + " not a leaf but no children_arcs");

      for (j = 0; j < children_arcs.size(); j++) {
        if (j < children_arcs.size() - 1)
          v +=
              children_arcs
                  .elementAt(j)
                  .monotonic_end
                  .display(
                      dum, children_arcs.elementAt(j).path_from_start_to_end(), show_embeddings);
        else
          v +=
              children_arcs
                  .elementAt(j)
                  .monotonic_end
                  .display(
                      indexes,
                      children_arcs.elementAt(j).path_from_start_to_end(),
                      show_embeddings);
      }
    }

    return v;
  }

  public String toString() {
    String v = "[" + depth + "><" + degree + "] ",
        classification = "(" + DF.format(alpha_value) + ")";
    int nblobbed = 0;
    DecisionTreeNode dn;
    Iterator<DecisionTreeNode> it;

    if (handle.name != 0) v += "[#" + handle.name + "]";
    else v += "[#0:root]";

    if (is_leaf) v += " leaf " + classification;
    else {

      if ((leaves_blobbed != null) && (leaves_blobbed.size() > 0)) {
        it = leaves_blobbed.iterator();

        v += " {";
        while (it.hasNext()) {
          if (nblobbed > 0) v += ",";
          dn = it.next();
          v += "#" + dn.name;
          nblobbed++;
        }
        v += "}";
      }
      v += " internal " + classification;
    }

    return v;
  }
}

class DecisionTreeSkipTreeArc {

  public static String USE_CARDINALS = "USE_CARDINALS",
      USE_BOOSTING_WEIGHTS = "USE_BOOSTING_WEIGHTS";

  public static String[] ALL_ALPHA_TYPES = {USE_CARDINALS, USE_BOOSTING_WEIGHTS};

  DecisionTreeNode start, end;
  MonotonicTreeNode monotonic_start, monotonic_end;

  boolean[] path_from_start_to_end;

  // provides the path to go from start to end, on the form of telling, from start, which of the
  // left (true) or right (false) neighbor is to be used
  // end is deeper than start in a DecisionTree, but is not necessarily a leaf

  DecisionTreeSkipTreeArc(MonotonicTreeNode st, MonotonicTreeNode en) {
    monotonic_start = st;
    monotonic_end = en;
  }

  DecisionTreeSkipTreeArc(DecisionTreeNode sn, Vector<Boolean> boolvals, DecisionTreeNode en) {
    if ((boolvals == null) || (boolvals.size() == 0))
      Dataset.perror("TreeArc.class :: cannot create arc between " + sn + " and " + en);

    start = sn;
    end = en;
    monotonic_start = monotonic_end = null;
    path_from_start_to_end = new boolean[boolvals.size()];
    int i;
    for (i = 0; i < boolvals.size(); i++)
      path_from_start_to_end[i] = boolvals.elementAt(i).booleanValue();
  }

  public String path_from_start_to_end() {
    if (path_from_start_to_end == null) return "null";
    else {
      String val = "(";
      int i;

      for (i = 0; i < path_from_start_to_end.length; i++)
        val += (path_from_start_to_end[i] ? "1" : "0");
      val += ")";
      return val;
    }
  }

  public String toString() {
    String ret = start.name + " => ";
    int i;
    for (i = 0; i < path_from_start_to_end.length; i++)
      ret += (path_from_start_to_end[i] ? "1" : "0");
    ret += " => " + end.name;
    return ret;
  }

  public static void ALL_TREE_ARCS_FROM_DECISION_TREE(
      String which_alpha,
      Vector<DecisionTreeSkipTreeArc> current_set_of_arcs_in_MonotonicTreeGraph,
      Vector<DecisionTreeSkipTreeArc> current_set_of_TreeNode_to_Blobbed_leaves,
      DecisionTreeNode current_start,
      Vector<Boolean> current_boolvals,
      DecisionTreeNode current_end) {
    double value_current_end = -1.0, value_current_start = -1.0;
    if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_CARDINALS)) {
      value_current_end = current_end.node_prediction_from_cardinals;
      value_current_start = current_start.node_prediction_from_cardinals;
    } else if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_BOOSTING_WEIGHTS)) {
      value_current_end = current_end.node_prediction_from_boosting_weights;
      value_current_start = current_start.node_prediction_from_boosting_weights;
    } else Dataset.perror("MonotonicTreeGraph.class :: no such prediction as " + which_alpha);

    if (Math.abs(value_current_end) > Math.abs(value_current_start)) {
      // new TreeArc !
      current_set_of_arcs_in_MonotonicTreeGraph.addElement(
          new DecisionTreeSkipTreeArc(current_start, current_boolvals, current_end));

      if (!current_end.is_leaf) {
        // start again from the children of current_end

        Vector<Boolean> left_booleans = new Vector<>(), right_booleans = new Vector<>();
        left_booleans.addElement(new Boolean(true));
        right_booleans.addElement(new Boolean(false));
        ALL_TREE_ARCS_FROM_DECISION_TREE(
            which_alpha,
            current_set_of_arcs_in_MonotonicTreeGraph,
            current_set_of_TreeNode_to_Blobbed_leaves,
            current_end,
            left_booleans,
            current_end.left_child);
        ALL_TREE_ARCS_FROM_DECISION_TREE(
            which_alpha,
            current_set_of_arcs_in_MonotonicTreeGraph,
            current_set_of_TreeNode_to_Blobbed_leaves,
            current_end,
            right_booleans,
            current_end.right_child);
      }
    } else if (current_end.is_leaf) {
      // Blobbed leaf: does not appear in the hyperbolic representation

      current_set_of_TreeNode_to_Blobbed_leaves.addElement(
          new DecisionTreeSkipTreeArc(current_start, current_boolvals, current_end));
    } else {
      // tries again with same start and the children of current_end

      Vector<Boolean> left_booleans = new Vector<>(current_boolvals),
          right_booleans = new Vector<>(current_boolvals);
      left_booleans.addElement(new Boolean(true));
      right_booleans.addElement(new Boolean(false));
      ALL_TREE_ARCS_FROM_DECISION_TREE(
          which_alpha,
          current_set_of_arcs_in_MonotonicTreeGraph,
          current_set_of_TreeNode_to_Blobbed_leaves,
          current_start,
          left_booleans,
          current_end.left_child);
      ALL_TREE_ARCS_FROM_DECISION_TREE(
          which_alpha,
          current_set_of_arcs_in_MonotonicTreeGraph,
          current_set_of_TreeNode_to_Blobbed_leaves,
          current_start,
          right_booleans,
          current_end.right_child);
    }
  }

  public static Vector<Vector<DecisionTreeSkipTreeArc>> ALL_TREE_ARCS_FROM_DECISION_TREE(
      String which_alpha, DecisionTree dt) {
    DecisionTreeNode r = dt.root;
    Vector<DecisionTreeSkipTreeArc> current_set_of_arcs_in_MonotonicTreeGraph = new Vector<>();
    Vector<DecisionTreeSkipTreeArc> current_set_of_TreeNode_to_Blobbed_leaves = new Vector<>();

    if (!r.is_leaf) {
      Vector<Boolean> left_booleans = new Vector<>();
      Vector<Boolean> right_booleans = new Vector<>();
      left_booleans.addElement(new Boolean(true));
      right_booleans.addElement(new Boolean(false));
      ALL_TREE_ARCS_FROM_DECISION_TREE(
          which_alpha,
          current_set_of_arcs_in_MonotonicTreeGraph,
          current_set_of_TreeNode_to_Blobbed_leaves,
          r,
          left_booleans,
          r.left_child);
      ALL_TREE_ARCS_FROM_DECISION_TREE(
          which_alpha,
          current_set_of_arcs_in_MonotonicTreeGraph,
          current_set_of_TreeNode_to_Blobbed_leaves,
          r,
          right_booleans,
          r.right_child);
    }

    Vector<Vector<DecisionTreeSkipTreeArc>> ret = new Vector<>();
    ret.addElement(current_set_of_arcs_in_MonotonicTreeGraph);
    ret.addElement(current_set_of_TreeNode_to_Blobbed_leaves);

    return ret;
  }
}

public class MonotonicTreeGraph implements Debuggable {

  public static boolean USE_SARKAR_SCALING = false;
  // if true, scales hyperbolic distances by distance_scaling

  DecisionTree handle;

  MonotonicTreeNode root;

  Vector<MonotonicTreeNode> leaves;

  String which_alpha;

  int depth, number_internal_nodes, max_degree_internal_node;

  private double distance_scaling;
  // Sarkar's "scaling" for embeddings with guaranteed HyperbolicPoint2D.SARKAR_EPSILON
  // approximation at the leaves
  // enforces test to use it

  double expected_embedding_quality_error;

  boolean Sarkar_ready;

  MonotonicTreeGraph(String wa, DecisionTree dt) {
    which_alpha = wa;
    handle = dt;
    depth = max_degree_internal_node = -1;
    number_internal_nodes = 0;
    leaves = null;
    Sarkar_ready = false;
    expected_embedding_quality_error = -1.0;
  }

  public static int[] PROCESS_TREE_GRAPHS(String which_alpha, Algorithm a) {
    int i, j, k, last_index_plotted = -1;
    int[] ret = null;
    DecisionTree dt;
    MonotonicTreeGraph tg;
    Vector<Vector<DecisionTreeSkipTreeArc>> all_tree_arcs_from_decision_tree;

    Vector<DecisionTreeSkipTreeArc> current_set_of_arcs_in_MonotonicTreeGraph;
    Vector<DecisionTreeSkipTreeArc> current_set_of_TreeNode_to_Blobbed_leaves;

    DecisionTree.DISPLAY_INTERNAL_NODES_CLASSIFICATION = true;

    for (i = 0; i < a.all_algorithms.size(); i++) {
      if (a.all_algorithms.elementAt(i).name.equals((Boost.KEY_NAME_LOG_LOSS))) {
        last_index_plotted = i;
        for (j = 0; j < a.all_algorithms.elementAt(i).recordAllTrees.length; j++) { // nb CV

          if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_CARDINALS))
            a.all_algorithms.elementAt(i).recordAllMonotonicTreeGraphs_cardinals[j] =
                new MonotonicTreeGraph[a.all_algorithms.elementAt(i).recordAllTrees[j].length];
          else if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_BOOSTING_WEIGHTS))
            a.all_algorithms.elementAt(i).recordAllMonotonicTreeGraphs_boosting_weights[j] =
                new MonotonicTreeGraph[a.all_algorithms.elementAt(i).recordAllTrees[j].length];
          else Dataset.perror("MonotonicTreeGraph.class :: no such prediction as " + which_alpha);

          for (k = 0; k < a.all_algorithms.elementAt(i).recordAllTrees[j].length; k++) {

            dt = a.all_algorithms.elementAt(i).recordAllTrees[j][k];
            tg = new MonotonicTreeGraph(which_alpha, dt);
            tg.root = new MonotonicTreeNode(which_alpha, dt.root, 0);
            tg.root.is_root = true;

            all_tree_arcs_from_decision_tree =
                DecisionTreeSkipTreeArc.ALL_TREE_ARCS_FROM_DECISION_TREE(which_alpha, tg.handle);

            current_set_of_arcs_in_MonotonicTreeGraph =
                all_tree_arcs_from_decision_tree.elementAt(0);
            current_set_of_TreeNode_to_Blobbed_leaves =
                all_tree_arcs_from_decision_tree.elementAt(1);

            if (current_set_of_arcs_in_MonotonicTreeGraph.size() > 0)
              tg.number_internal_nodes = 1; // to make sure to include the root

            tg.buildFromSkipTreeArcs(
                which_alpha,
                tg.root,
                current_set_of_arcs_in_MonotonicTreeGraph,
                current_set_of_TreeNode_to_Blobbed_leaves);
            tg.distance_scaling =
                ((1.0 + HyperbolicPoint2D.SARKAR_EPSILON) / HyperbolicPoint2D.SARKAR_EPSILON)
                    * Math.log(2.0 * ((double) tg.max_degree_internal_node) / Math.PI);
            tg.Sarkar_ready = true;

            tg.toWideEmbedding();

            if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_CARDINALS))
              a.all_algorithms.elementAt(i).recordAllMonotonicTreeGraphs_cardinals[j][k] = tg;
            else if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_BOOSTING_WEIGHTS))
              a.all_algorithms.elementAt(i).recordAllMonotonicTreeGraphs_boosting_weights[j][k] =
                  tg;
            else Dataset.perror("MonotonicTreeGraph.class :: no such prediction as " + which_alpha);

            current_set_of_arcs_in_MonotonicTreeGraph =
                current_set_of_TreeNode_to_Blobbed_leaves = null;
          }
        }
      } else {
        if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_CARDINALS))
          for (j = 0; j < a.all_algorithms.elementAt(i).recordAllTrees.length; j++)
            a.all_algorithms.elementAt(i).recordAllMonotonicTreeGraphs_cardinals[j] =
                new MonotonicTreeGraph[a.all_algorithms.elementAt(i).recordAllTrees[j].length];
        else if (which_alpha.equals(DecisionTreeSkipTreeArc.USE_BOOSTING_WEIGHTS))
          for (j = 0; j < a.all_algorithms.elementAt(i).recordAllTrees.length; j++)
            a.all_algorithms.elementAt(i).recordAllMonotonicTreeGraphs_boosting_weights[j] =
                new MonotonicTreeGraph[a.all_algorithms.elementAt(i).recordAllTrees[j].length];
        else Dataset.perror("MonotonicTreeGraph.class :: no such prediction as " + which_alpha);
      }
    }

    // sets the tree(s) to be plotted
    if (last_index_plotted > -1) {
      ret = new int[3];

      ret[0] = last_index_plotted;
      ret[1] = NUMBER_STRATIFIED_CV - 1;
      ret[2] = 0;
    }
    return ret;
  }

  public double distance_scaling() {
    if (MonotonicTreeGraph.USE_SARKAR_SCALING) return distance_scaling;
    else return 1.0;
  }

  public void toWideEmbedding() {
    if (!Sarkar_ready)
      Dataset.perror(
          "PoincareDiskEmbeddingUI.class :: cannot proceed to Wide embedding, monotonic tree graph"
              + " not finished");

    root.embedding_coordinates = HyperbolicPoint2D.POINCARE_ROOT(root);

    Vector<List<MonotonicTreeNode>> all_arcs = new Vector<>();
    List<MonotonicTreeNode> duml;

    // creates a fake first root because mtg's root does not map to the origin
    MonotonicTreeNode fake_root = MonotonicTreeNode.FAKE_NODE(), parent, child, grandchild;
    fake_root.embedding_coordinates = new HyperbolicPoint2D(0.0, 0.0);

    HyperbolicPoint2D start_pt, end_pt;

    // do it !
    double theta_parent,
        theta,
        angle_grandchild,
        deg,
        r,
        check_dist,
        initial_fan_angle = INITIAL_FAN_ANGLE,
        delta_angle,
        fan_per_grandchild,
        grandchild_initial_fan_angle,
        norm,
        norm_sq,
        error_ratio,
        expected_error_ratio = 0.0;
    // expected_error_ratio = 0.0 bcs no error to embed root;

    int i,
        number_plotted = 1,
        number_leaves_child,
        number_leaves_grandchild,
        number_leaves_covered = 0;
    int[] test_count;

    // Step 1: computes all fans
    root.fan_for_plotting = initial_fan_angle;

    duml = new ArrayList<>();
    duml.add(fake_root);
    duml.add(root);
    all_arcs.add(duml);

    while (all_arcs.size() > 0) {
      duml = all_arcs.remove(0);
      parent = duml.get(0);
      child = duml.get(1);

      if (child.fan_for_plotting == -1.0)
        Dataset.perror("MonotonicTreeGraph.class :: fan not computed for " + child);

      if (child.embedding_coordinates == null)
        Dataset.perror("MonotonicTreeGraph.class :: embedding not computed for " + child);

      // how many leaves
      test_count = new int[1];
      child.count_subleaves(test_count);
      number_leaves_child = test_count[0];

      // fan per grandchild
      fan_per_grandchild = child.fan_for_plotting / ((double) number_leaves_child);

      // starting angle
      delta_angle = Math.PI - (child.fan_for_plotting / 2.0);
      theta_parent =
          parent.embedding_coordinates.parent_angle_after_inversion(child.embedding_coordinates);
      theta_parent += delta_angle;

      angle_grandchild = theta_parent;

      number_leaves_covered = 0;
      for (i = 0; i < child.degree; i++) {
        grandchild = child.children_arcs.elementAt(i).monotonic_end;

        test_count = new int[1];
        grandchild.count_subleaves(test_count);
        number_leaves_grandchild = test_count[0];

        angle_grandchild += ((double) number_leaves_grandchild + 1) * fan_per_grandchild / 2.0;
        norm = child.embedded_radius_from_distance(grandchild, this); // BEWARE 2.0

        grandchild.embedding_coordinates =
            PoincareDiskEmbeddingUI.POINCARE_CIRCLE_INVERSION_ORIGIN_MODELE(
                child.embedding_coordinates,
                new HyperbolicPoint2D(
                    norm * Math.cos(angle_grandchild), norm * Math.sin(angle_grandchild)));

        // safe check distances match, REMOVE later
        check_dist = grandchild.distance_from_embedded_points(child, this);
        if (!Statistics.APPROXIMATELY_EQUAL(
            check_dist, Math.abs(grandchild.alpha_value) - Math.abs(child.alpha_value), EPS2))
          Dataset.perror(
              "MonotonicTreeGraph.class :: distance computation mismatch ("
                  + check_dist
                  + " != "
                  + (Math.abs(grandchild.alpha_value) - Math.abs(child.alpha_value))
                  + ")");

        angle_grandchild += ((double) number_leaves_grandchild + 1) * fan_per_grandchild / 2.0;

        norm_sq = grandchild.embedding_coordinates.euclid_norm_squared();
        error_ratio =
            100.0
                * (Math.abs(grandchild.alpha_value)
                    - HyperbolicPoint2D.ACOSH(1.0 + 2.0 * norm_sq / (1.0 - norm_sq)))
                / Math.abs(grandchild.alpha_value);

        if (Math.abs(error_ratio) < EPS2) error_ratio = 0.0;

        number_plotted++;
        expected_error_ratio += error_ratio;

        if (error_ratio < 0.0)
          Dataset.perror("MonotonicTreeGraph.class :: negative error " + error_ratio);

        if (!grandchild.is_leaf) {
          // computes grandchild fan

          grandchild.fan_for_plotting =
              GRANDCHILD_FAN_RATIO
                  * ((double) number_leaves_grandchild + 1)
                  * fan_per_grandchild; // approximation

          number_leaves_covered += number_leaves_grandchild;

          duml = new ArrayList<>();
          duml.add(child);
          duml.add(grandchild);
          all_arcs.add(duml);
        }
      }
    }

    expected_error_ratio /= (double) number_plotted;
    expected_embedding_quality_error = expected_error_ratio;
  }

  public void toSarkarEmbedding() {
    if (!Sarkar_ready)
      Dataset.perror(
          "PoincareDiskEmbeddingUI.class :: cannot proceed to Sarkar embedding, monotonic tree"
              + " graph not finished");

    root.embedding_coordinates = HyperbolicPoint2D.POINCARE_ROOT(root);

    Vector<List<MonotonicTreeNode>> all_arcs = new Vector<>();
    List<MonotonicTreeNode> duml;

    // creates a fake first root because mtg's root does not map to the origin
    MonotonicTreeNode fake_root = MonotonicTreeNode.FAKE_NODE(), parent, child, grandchild;
    fake_root.embedding_coordinates = new HyperbolicPoint2D(0.0, 0.0);
    duml = new ArrayList<>();
    duml.add(fake_root);
    duml.add(root);
    all_arcs.add(duml);

    // do it !
    double theta_parent, theta, angle, deg, r, check_dist, fan_angle = Math.PI / 2.0, delta_angle;
    int i, number_plotted = 0;

    while (all_arcs.size() > 0) {
      duml = all_arcs.remove(0);
      parent = duml.get(0);
      child = duml.get(1);

      delta_angle = Math.PI - (fan_angle / 2.0);
      theta_parent =
          parent.embedding_coordinates.parent_angle_after_inversion(child.embedding_coordinates);

      if (number_plotted == 0)
        System.out.println(
            "C angle(O, #" + child.handle.name + ") = " + Math.toDegrees(theta_parent));
      else
        System.out.println(
            "C angle(#"
                + parent.handle.name
                + ", #"
                + child.handle.name
                + ") = "
                + Math.toDegrees(theta_parent));
      number_plotted++;

      theta_parent += delta_angle;

      deg = (double) child.degree;
      if ((deg == 0.0) || (child.degree != child.children_arcs.size()))
        Dataset.perror(
            "MonotonicTreeGraph.class :: embedding a child with degree 0 OR degree mismatch for arc"
                + " ("
                + parent
                + " => "
                + child
                + ")");

      if (deg == 1.0) theta = fan_angle / (deg + 1.0);
      else {
        theta = fan_angle / (deg - 1.0);
        theta_parent -= theta;
      }

      angle = theta_parent;

      for (i = 0; i < child.degree; i++) {
        grandchild = child.children_arcs.elementAt(i).monotonic_end;

        if (Math.abs(child.alpha_value) > Math.abs(grandchild.alpha_value))
          Dataset.perror(
              "MonotonicTreeGraph.class :: bad ordering for distances wrt alpha values (child = "
                  + child
                  + ", grandchild = "
                  + grandchild
                  + ")");

        r = child.embedded_radius_from_distance(grandchild, this);

        angle += theta;
        grandchild.embedding_coordinates =
            PoincareDiskEmbeddingUI.POINCARE_CIRCLE_INVERSION_ORIGIN_MODELE(
                child.embedding_coordinates,
                new HyperbolicPoint2D(r * Math.cos(angle), r * Math.sin(angle)));

        // safe check distances match, REMOVE later
        check_dist = grandchild.distance_from_embedded_points(child, this);
        if (!Statistics.APPROXIMATELY_EQUAL(
            check_dist, Math.abs(grandchild.alpha_value) - Math.abs(child.alpha_value), EPS2))
          Dataset.perror(
              "MonotonicTreeGraph.class :: distance computation mismatch ("
                  + check_dist
                  + " != "
                  + (Math.abs(grandchild.alpha_value) - Math.abs(child.alpha_value))
                  + ")");

        if (!grandchild.is_leaf) {
          duml = new ArrayList<>();
          duml.add(child);
          duml.add(grandchild);
          all_arcs.add(duml);
        }
      }
    }
  }

  public String toString() {
    int i;
    String v =
        "MonotonicTreeGraph ["
            + which_alpha
            + "] (name = #"
            + handle.name
            + " | depth = "
            + depth
            + " | max degree = "
            + max_degree_internal_node
            + " | #nodes = [I:"
            + number_internal_nodes
            + ";L:"
            + ((leaves != null) ? leaves.size() : "null")
            + "])\n";
    MonotonicTreeNode dumn;

    v += root.display(new HashSet<Integer>(), "()", Sarkar_ready);

    v += "Leaves:";

    Iterator it = leaves.iterator();
    while (it.hasNext()) {
      v += " ";
      dumn = (MonotonicTreeNode) it.next();
      v += "#" + dumn.handle.name;
    }
    v += ".\n";

    return v;
  }

  public void buildFromSkipTreeArcs(
      String which_alpha,
      MonotonicTreeNode currentNode,
      Vector<DecisionTreeSkipTreeArc> current_set_of_arcs_in_MonotonicTreeGraph,
      Vector<DecisionTreeSkipTreeArc> current_set_of_TreeNode_to_Blobbed_leaves) {
    // currentNode *must have been lightly instanciated (just handle) before calling this method*
    // fills its variables, removes the corresponding stuff from current_set_of_* and calls again
    // the method with children lightly instanciated, if necessary
    // also completes variables monotonic_start, monotonic_end in DecisionTreeSkipTreeArcs
    // (monotonic_end only if end not a leaf)

    int i;
    DecisionTreeSkipTreeArc dtsa;
    boolean bd;
    Vector<MonotonicTreeNode> next_currentNode = null;

    // blobbed leaves
    if ((current_set_of_TreeNode_to_Blobbed_leaves != null)
        && (current_set_of_TreeNode_to_Blobbed_leaves.size() > 0)) {
      i = 0;
      do {
        dtsa = current_set_of_TreeNode_to_Blobbed_leaves.elementAt(i);
        if (dtsa.start.equals(currentNode.handle)) {
          // adds new blobbed leaves to currentNode
          if (currentNode.leaves_blobbed == null) currentNode.leaves_blobbed = new HashSet<>();

          bd =
              currentNode.leaves_blobbed.add(
                  current_set_of_TreeNode_to_Blobbed_leaves.elementAt(i).end);
          current_set_of_TreeNode_to_Blobbed_leaves.removeElementAt(i);
          // just discard the DecisionTreeSkipTreeArc

        } else i++;
      } while (i < current_set_of_TreeNode_to_Blobbed_leaves.size());
    }

    if ((current_set_of_arcs_in_MonotonicTreeGraph != null)
        && (current_set_of_arcs_in_MonotonicTreeGraph.size() > 0)) {
      i = 0;
      do {
        dtsa = current_set_of_arcs_in_MonotonicTreeGraph.elementAt(i);
        if (dtsa.start.equals(currentNode.handle)) {
          dtsa.monotonic_start = currentNode;

          if (currentNode.children_arcs == null) currentNode.children_arcs = new Vector<>();

          if (dtsa.monotonic_start.depth + 1 > depth) depth = dtsa.monotonic_start.depth + 1;

          dtsa.monotonic_end =
              new MonotonicTreeNode(which_alpha, dtsa.end, dtsa.monotonic_start.depth + 1);

          if (dtsa.end.is_leaf) {
            dtsa.monotonic_end.is_leaf();

            if (leaves == null) leaves = new Vector<>();
            leaves.addElement(dtsa.monotonic_end);
          } else {

            if (next_currentNode == null) next_currentNode = new Vector<>();

            if (next_currentNode.contains(dtsa.monotonic_end))
              Dataset.perror(
                  "MonotonicTreeGraph.class :: MonotonicTreeNode "
                      + dtsa.monotonic_end
                      + " already present in next_currentNode");
            else {
              next_currentNode.addElement(dtsa.monotonic_end);
            }
            number_internal_nodes++;
          }
          currentNode.children_arcs.addElement(dtsa);
          currentNode.degree++;
          if ((max_degree_internal_node == -1) || (currentNode.degree > max_degree_internal_node))
            max_degree_internal_node = currentNode.degree;

          current_set_of_arcs_in_MonotonicTreeGraph.removeElementAt(i);
        } else i++;
      } while (i < current_set_of_arcs_in_MonotonicTreeGraph.size());
    }

    if ((next_currentNode != null) && (next_currentNode.size() > 0))
      for (i = 0; i < next_currentNode.size(); i++)
        buildFromSkipTreeArcs(
            which_alpha,
            next_currentNode.elementAt(i),
            current_set_of_arcs_in_MonotonicTreeGraph,
            current_set_of_TreeNode_to_Blobbed_leaves);
  }
}
