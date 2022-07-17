// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Generator_Node
 *****/

class Generator_Arc implements Debuggable {
  // (p,q) + feature

  int name;

  Generator_Node begin_node, end_node;
  Feature feature_arc;
  int feature_arc_index;
  // handle on the feature index in
  // [begin_node|end_node].myGeneratorTree.myBoost.myDomain.myDS.features
  // must match begin_node.feature_node_index (just for checking consistency)

  Generator_Arc(Generator_Node b, Generator_Node e, Feature f, int fni, int n) {
    name = n;
    if ((b == null) || (e == null) || (f == null))
      Dataset.perror("Generator_Edge.class :: null values for constructor");
    begin_node = b;
    end_node = e;
    feature_arc = Feature.copyOf(f, false); // stores a copy
    feature_arc_index = fni;
  }

  public String toString() {
    String s = "";
    s += " [ " + feature_arc.toStringInTree(true, false) + " ] ";
    return s;
  }
}

class Generator_Node implements Debuggable {
  public static int OUT_DEGREE = 2;
  // out-degree of any non-leaf generator node

  int name, depth;
  // index of the node in the GT, the root is labeled 0; depth = depth in the tree (0 = root);

  boolean is_leaf;
  // if true, sampling leaf => feature_node_index = -1, children_arcs = null, multi_p = null,
  // multi_tau = null;
  double p_node; // probability to reach the node (sum over leaves of any rooted subtree = 1.0)

  int feature_node_index;
  // handle on the feature index, just a repeat of the OUTGOING arcs for checking
  Feature feature_node;
  // copy of the feature domain available at the node.

  Generator_Arc[] children_arcs; // handles on children nodes
  // Constraint: each is the *same* feature and the union of all modalities = the local domain of
  // the feature
  double[] multi_p; // multinomial probabilities

  // Feature information at the node level
  Generator_Tree myGeneratorTree;

  Generator_Node myParentGenerator_Node;
  int myParentGenerator_Node_children_number;
  // used to traverse the tree: myParentGenerator_Node_children_number is the #index in
  // myParentGenerator_Node.children_arcs, useful to traverse the generator

  Vector<Feature> all_features_domain;

  public double node_volume() {
    int i;
    double vol = 1.0;
    for (i = 0; i < myGeneratorTree.myBoost.myDomain.myDS.number_domain_features(); i++)
      vol *= all_features_domain.elementAt(i).length();
    return vol;
  }

  public void recursive_write_nodes(FileWriter f) throws IOException {
    int i;
    f.write(string_node_save() + "\n");
    if (!is_leaf)
      for (i = 0; i < children_arcs.length; i++) children_arcs[i].end_node.recursive_write_nodes(f);
  }

  public void compute_all_features_domain() {
    all_features_domain =
        Generator_Node.ALL_FEATURES_DOMAINS_AT_SAMPLING_NODE(myGeneratorTree, this);
  }

  public void recursive_fill_node_counts(int[] node_counts, String[] features_names_from_file) {
    boolean found = false;
    int i = 0;

    if (!is_leaf) {
      do {
        if (feature_node.name.equals(features_names_from_file[i])) found = true;
        else i++;
      } while ((!found) && (i < features_names_from_file.length));
      if (!found)
        Dataset.perror(
            "Generator_Node.class :: node feature name "
                + feature_node.name
                + " not in the list of features");
      node_counts[i]++;

      for (i = 0; i < children_arcs.length; i++)
        children_arcs[i].end_node.recursive_fill_node_counts(node_counts, features_names_from_file);
    }
  }

  public String string_node_save() {
    // provides a simple line saving the node in the context of its gt being saved
    String ret = "";
    int i;
    ret = name + "\t";
    if (myParentGenerator_Node == null) // root
    ret += "-1\t-1";
    else ret += myParentGenerator_Node.name + "\t" + myParentGenerator_Node_children_number;
    ret += "\t";

    ret += depth + "\t" + p_node + "\t" + is_leaf;
    if (!is_leaf) {
      ret += "\t" + feature_node_index + "\t" + feature_node.name + "\t";
      for (i = 0; i < multi_p.length; i++) {
        ret += multi_p[i] + "\t" + children_arcs[i].end_node.name;
        if (i < multi_p.length - 1) ret += "\t";
      }
    }
    return ret;
  }

  public void recursive_write_arcs(FileWriter f) throws IOException {
    int i;
    if (!is_leaf) {
      for (i = 0; i < children_arcs.length; i++) {
        f.write(
            name
                + "\t"
                + children_arcs[i].end_node.name
                + "\t"
                + children_arcs[i].feature_arc_index
                + "\t"
                + children_arcs[i].feature_arc.name
                + "\t"
                + Feature.SAVE_FEATURE(children_arcs[i].feature_arc)
                + "\n");
        children_arcs[i].end_node.recursive_write_arcs(f);
      }
    }
  }

  public void check_names(HashSet<Integer> h) {
    // recursively checks that names do not appear twice in the GT (in case they have been changed
    // such as for copycat training)
    int i;

    if (h.contains(new Integer(name)))
      Dataset.perror("Generator_Node.class :: name " + name + " already in GT -- naming problem");
    if (!is_leaf) {
      h.add(new Integer(name));
      for (i = 0; i < children_arcs.length; i++) children_arcs[i].end_node.check_names(h);
    }
  }

  public static Vector<Feature> ALL_FEATURES_DOMAINS_AT_SAMPLING_NODE(
      Generator_Tree gt, Generator_Node gn) {
    // returns a vector of features stating the available domains given the position of the
    // generator node gn in gt
    // for the tree's root, this would just be all features with all domains;

    Vector<Feature> v = new Vector<Feature>();
    Vector<Generator_Arc> seq = new Vector<Generator_Arc>();
    Generator_Arc ga;

    int i;
    for (i = 0; i < gt.myBoost.myDomain.myDS.number_domain_features(); i++)
      v.addElement(Generator_Node.FEATURE_DOMAIN_AT_SAMPLING_NODE(gt, gn, i));

    return v;
  }

  public static Feature FEATURE_DOMAIN_AT_SAMPLING_NODE(
      Generator_Tree gt, Generator_Node gnori, int fni) {
    // returns feature indexed fni stating the available domains given the position of the generator
    // node gn in gt
    // for the tree's root, this would just be the initial feature

    Feature f =
        Feature.copyOf(
            (Feature) gt.myBoost.myDomain.myDS.domain_feature(fni),
            true); // Feature.copyOf((Feature) gt.myBoost.myDomain.myDS.features.elementAt(fni),
    // true);
    if (gt.has_root(gnori)) return f;
    Vector<Generator_Arc> seq = new Vector<Generator_Arc>();
    Generator_Arc ga;
    Generator_Node gn = gnori;
    int i;

    // must be sampling node
    if (!gnori.is_leaf)
      Dataset.perror("Generator_Node.class :: attempting to split a non sampling node");

    while (gn.depth > 0) {
      seq.addElement(
          gn.myParentGenerator_Node
              .children_arcs[
              gn.myParentGenerator_Node_children_number]); // adds the arc whose end is gn
      gn = gn.myParentGenerator_Node;
    }

    if (seq.size() > 0) {
      Collections.reverse(seq); // puts edges first
      for (i = 0; i < seq.size(); i++) {
        ga = seq.elementAt(i);
        if (ga.feature_arc_index == fni) {
          if (!Feature.IS_SUBFEATURE(ga.feature_arc, ga.feature_arc_index, f, fni))
            Dataset.perror(
                "Generator_Node.class :: feature " + ga.feature_arc + " not a subfeature of " + f);
          f = Feature.copyOf(ga.feature_arc, true);
        }
      }
    }

    seq = null;

    return f;
  }

  public static boolean LEAF_COULD_GENERATE_EXAMPLE(Generator_Node gn, Example e) {
    if (!gn.is_leaf) Dataset.perror("Generator_Node.class :: node not a leaf");
    int i, ei;
    double ed;
    String es;

    if (gn.all_features_domain == null)
      Dataset.perror(
          "Generator_Node.class :: GT leaf " + gn.name + " has all_features_domain == null");

    Vector<Feature> v = gn.all_features_domain;

    Feature f;
    boolean found = false;
    for (i = 0; i < v.size(); i++) {
      f = v.elementAt(i);
      found = false;

      if (Unknown_Feature_Value.IS_UNKNOWN(
          (Object) e.typed_features.elementAt(i))) { // WARNING: Unknown always considered generable
        found = true;
      } else if (Feature.IS_CONTINUOUS(f.type)) {
        if (!e.typed_features.elementAt(i).getClass().getSimpleName().equals("Double"))
          Dataset.perror(
              "Generator_Node.class :: class mismatch for double value as e["
                  + i
                  + "] = "
                  + e.typed_features.elementAt(i)
                  + " for example "
                  + e);
        ed = ((Double) e.typed_features.elementAt(i)).doubleValue();
        if ((f.dmin <= ed) && (f.dmax >= ed)) found = true;
      } else if (Feature.IS_INTEGER(f.type)) {
        if (!e.typed_features.elementAt(i).getClass().getSimpleName().equals("Integer"))
          Dataset.perror(
              "Generator_Node.class :: class mismatch for int value as e["
                  + i
                  + "] = "
                  + e.typed_features.elementAt(i)
                  + " for example "
                  + e);
        ei = ((Integer) e.typed_features.elementAt(i)).intValue();
        if ((f.imin <= ei) && (f.imax >= ei)) found = true;
      } else if (Feature.IS_NOMINAL(f.type)) {
        if (!e.typed_features.elementAt(i).getClass().getSimpleName().equals("String"))
          Dataset.perror(
              "Generator_Node.class :: class mismatch for String value as e["
                  + i
                  + "] = "
                  + e.typed_features.elementAt(i)
                  + " for example "
                  + e);
        es = (String) e.typed_features.elementAt(i);
        if (f.modalities.contains(es)) found = true;
      }
      if (!found) return false;
    }
    return true;
  }

  Generator_Node(Generator_Tree gt, Generator_Node pn, int cn) {
    myGeneratorTree = gt;
    myParentGenerator_Node = pn;
    myParentGenerator_Node_children_number = cn;

    name = depth = -1;

    children_arcs = null;
    feature_node = null;
    multi_p = null;

    is_leaf = true;
    feature_node_index = -1;
    p_node = -1.0;

    all_features_domain = null;
  }

  Generator_Node(Generator_Tree t, Generator_Node pn, int v, int d, int cn) {
    this(t, pn, cn);

    name = v;
    depth = d;
    if (depth == 0) p_node = 1.0; // useless to compute it
  }

  public boolean equals(Object o) {
    if (o == this) return true;
    if (!(o instanceof Generator_Node)) return false;
    Generator_Node test = (Generator_Node) o;
    if ((test.name == name)
        && (test.myGeneratorTree.name == myGeneratorTree.name)
        && (test.depth == depth)) return true;
    return false;
  }

  public String toString() {
    String v = "";

    if (name == 1) v += "\n[#1:root]";
    else {
      if (myParentGenerator_Node == null)
        Dataset.perror("Generator_Node.class :: non root node with null parent");
      if (myParentGenerator_Node_children_number == -1)
        Dataset.perror(
            "Generator_Node.class :: non root node with non null parent but -1 index in parents"
                + " children");

      v +=
          "["
              + DF4.format(myParentGenerator_Node.multi_p[myParentGenerator_Node_children_number])
              + ","
              + myParentGenerator_Node.children_arcs[myParentGenerator_Node_children_number]
              + "]--[#"
              + name;
      if (is_leaf) v += " (sampling)]";
      else v += "]";
    }
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

      for (i = 0; i < children_arcs.length - 1; i++) v += children_arcs[i].end_node.display(dum);

      v += children_arcs[children_arcs.length - 1].end_node.display(indexes);
    }

    return v;
  }

  public void recursive_copy(Generator_Node model, Generator_Node parent) {
    int i;
    Generator_Node dum;

    name = model.name;
    depth = model.depth;
    is_leaf = model.is_leaf;
    p_node = model.p_node;
    if (parent != null) myGeneratorTree = parent.myGeneratorTree;
    myParentGenerator_Node = parent;
    myParentGenerator_Node_children_number = model.myParentGenerator_Node_children_number;
    feature_node = null;
    feature_node_index = -1;

    if (((model.multi_p != null) && (model.children_arcs == null))
        || ((model.multi_p == null) && (model.children_arcs != null)))
      Dataset.perror(
          "Generator_Node.class :: inconsistencies in children parameters at node " + this);

    if (!is_leaf) {
      feature_node_index = model.feature_node_index;
      feature_node = Feature.copyOf(model.feature_node, true);

      multi_p = new double[model.multi_p.length];
      for (i = 0; i < multi_p.length; i++) multi_p[i] = model.multi_p[i];

      children_arcs = new Generator_Arc[model.children_arcs.length];

      for (i = 0; i < children_arcs.length; i++) {
        dum = new Generator_Node(myGeneratorTree, this, i);
        children_arcs[i] =
            new Generator_Arc(
                this,
                dum,
                Feature.copyOf(model.children_arcs[i].feature_arc, true),
                model.children_arcs[i].feature_arc_index,
                model.children_arcs[i].name);
      }

      for (i = 0; i < children_arcs.length; i++)
        children_arcs[i].end_node.recursive_copy(model.children_arcs[i].end_node, this);
    } else myGeneratorTree.leaves.add(this);
  }

  public Generator_Node[] split(
      int test_feature_node_index,
      Feature test_feature_node,
      double test_p_final,
      Vector<Feature> rv_element) {
    // just computes the split and returns new leaves
    Generator_Node leaf;
    int i;
    Generator_Node[] ret = new Generator_Node[Generator_Node.OUT_DEGREE];

    feature_node_index = test_feature_node_index;
    feature_node = test_feature_node;

    // creates the subtree: arcs + new leaves

    children_arcs = new Generator_Arc[Generator_Node.OUT_DEGREE];
    for (i = 0; i < Generator_Node.OUT_DEGREE; i++) {
      leaf =
          new Generator_Node(
              myGeneratorTree, this, myGeneratorTree.number_nodes + i + 1, depth + 1, i);
      children_arcs[i] =
          new Generator_Arc(
              this,
              leaf,
              rv_element.elementAt(i),
              feature_node_index,
              myGeneratorTree.number_nodes + i);
      ret[i] = leaf;
    }

    // more specific code just for binary splits
    multi_p = new double[2];
    multi_p[0] = 1.0 - test_p_final;
    multi_p[1] = test_p_final;

    ret[0].p_node = p_node * multi_p[0];
    ret[1].p_node = p_node * multi_p[1];

    is_leaf = false;

    if ((ret == null) || (ret.length == 1))
      Dataset.perror("Generator_Node.class :: less than two new leaves in generator");

    return ret;
  }

  public Example one_example(int id) {
    if (all_features_domain == null)
      Dataset.perror("Generator_Node.class :: GT leaf has all_features_domain == null");

    Vector<Feature> v = all_features_domain;
    Vector exval = new Vector();
    Example e;
    int i, ival;
    Feature f;
    double dval;
    String sval;

    for (i = 0; i < v.size(); i++) {
      f = (Feature) v.elementAt(i);
      if (f.type.equals(Feature.CONTINUOUS)) {
        dval = f.dmin + ((Algorithm.R.nextDouble()) * (f.dmax - f.dmin));
        exval.addElement(Double.toString(dval));
      } else if (f.type.equals(Feature.INTEGER)) {
        ival = f.imin + (Algorithm.R.nextInt(f.imax - f.imin + 1));
        exval.addElement(Integer.toString(ival));
      } else if (f.type.equals(Feature.NOMINAL))
        exval.addElement(
            new String((String) f.modalities.elementAt(Algorithm.R.nextInt(f.modalities.size()))));
    }

    e = new Example(id, exval, v, false, this);

    return e;
  }

  public void resample_feature_in_generated_example(Example e, int feat_index) {
    if (all_features_domain == null)
      Dataset.perror("Generator_Node.class :: GT leaf has all_features_domain == null");

    Vector<Feature> v = all_features_domain;
    Feature f = (Feature) v.elementAt(feat_index);
    double dval = -1.0;
    int ival = -1;
    String sval = "";
    if (f.type.equals(Feature.CONTINUOUS)) {
      dval = f.dmin + ((Algorithm.R.nextDouble()) * (f.dmax - f.dmin));
      e.typed_features.setElementAt(new Double(dval), feat_index);
    } else if (f.type.equals(Feature.INTEGER)) {
      ival = f.imin + (Algorithm.R.nextInt(f.imax - f.imin + 1));
      e.typed_features.setElementAt(new Integer(ival), feat_index);
    } else if (f.type.equals(Feature.NOMINAL)) {
      sval = (String) f.modalities.elementAt(Algorithm.R.nextInt(f.modalities.size()));
      e.typed_features.setElementAt(new String(sval), feat_index);
    }
    e.generating_leaf = this;
  }
}
