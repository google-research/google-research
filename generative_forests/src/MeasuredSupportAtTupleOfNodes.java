// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class MeasuredSupportAtTupleOfNodes and related classes
 * Mainly for GEOT
 *****/

class MeasuredSupportAtTupleOfNodes implements Debuggable{
  // used for GEOT
  // a support with subset of training observations (proportions) attached

  public static String LEFT_CHILD = "LEFT_CHILD",
      RIGHT_CHILD = "RIGHT_CHILD",
      NO_CHILD = "NO_CHILD";

    // one instance = (measured support, tuple of pointers on nodes in the trees where the measured support is created as the intersection of all measured supports)
    // all purpose data structure:
    // (i) for missing data imputation
    // (ii) for boosting training 
    
    public MeasuredSupport generative_support;

    Vector <Node> tree_nodes_support;
    // set of nodes whose support intersection makes the support
    // (i) missing data imputation: progressively flushed out (empty iff all nodes are leaves), then stored and used to impute
    // (ii) boosting: not flushed out
    
    GenerativeModelBasedOnEnsembleOfTrees geot;

    MeasuredSupportAtTupleOfNodes [] split_top_down_boosting_statistics;
    // 0/1 = left/right = output of split_for_boosting_computations
    // not copied, just to store boosting computations

    boolean marked_for_update;
    // FLAG used for boosting: after having updated the Hashset of supports, go through the (new) MeasuredSupportAtTupleOfNodes and updates the relevant old leaf with new child = leaf
    int marked_for_update_index_tree;
    String marked_for_update_which_child;
    MeasuredSupportAtTupleOfNodes marked_for_update_parent_msatol;

    public static void CHECK_BOOSTING_CONSISTENCY(GenerativeModelBasedOnEnsembleOfTrees g, HashSet <MeasuredSupportAtTupleOfNodes> tol){
	if ( (tol == null) || (tol.size() == 0) )
	    return;

	Iterator it = tol.iterator();
	MeasuredSupportAtTupleOfNodes ms;
	
	while(it.hasNext()){
	    ms = (MeasuredSupportAtTupleOfNodes) it.next();
      if (ms.tree_nodes_support.size() != g.trees.size())
        Dataset.perror(
            "MeasuredSupportAtTupleOfNodes.class :: inconsistent MeasuredSupportAtTupleOfNodes");
	}
    }

    MeasuredSupportAtTupleOfNodes(Support s, int [] observations_indexes){
	geot = null;
	split_top_down_boosting_statistics = null;
	marked_for_update = false;
	marked_for_update_parent_msatol = null;
	marked_for_update_which_child = null;
	marked_for_update_index_tree = -1;
	generative_support = new MeasuredSupport(s, observations_indexes);
    }

    MeasuredSupportAtTupleOfNodes(GenerativeModelBasedOnEnsembleOfTrees g, boolean filter_out_leaves, boolean check_all_are_leaves){
	// two uses
	// (i) initializes MDS to the whole domain of unknown features: filter_out_leaves = true, check_all_are_leaves = false
	// (ii) top down boosting: filter_out_leaves = false, check_all_are_leaves = true
	geot = g;

	// boosting stuff
	split_top_down_boosting_statistics = null;
	marked_for_update = false;
	marked_for_update_index_tree = -1;
	marked_for_update_parent_msatol = null;
	marked_for_update_which_child = MeasuredSupportAtTupleOfNodes.NO_CHILD;

	int i;

	tree_nodes_support = new Vector<>();
	for (i=0;i<geot.trees.size();i++){
      if ((check_all_are_leaves) && (!geot.trees.elementAt(i).root.is_leaf))
        Dataset.perror(
            "MeasuredSupportAtTupleOfNodes.class :: not all leaves ("
                + geot.trees.elementAt(i).root
                + ")");
	    if ( ( (filter_out_leaves) && (!geot.trees.elementAt(i).root.is_leaf) ) || (!filter_out_leaves) )
		tree_nodes_support.addElement(geot.trees.elementAt(i).root);
	}
    }

    MeasuredSupportAtTupleOfNodes(GenerativeModelBasedOnEnsembleOfTrees g, boolean filter_out_leaves, boolean check_all_are_leaves, int exception){
	this(g, filter_out_leaves, check_all_are_leaves);
	generative_support = new MeasuredSupport(g, exception);
    }

    
    public static MeasuredSupportAtTupleOfNodes copyOf(MeasuredSupportAtTupleOfNodes mds, boolean filter_out_leaves, boolean check_all_are_leaves, boolean mark_parent){
	MeasuredSupportAtTupleOfNodes ret = new MeasuredSupportAtTupleOfNodes(mds.geot, filter_out_leaves, check_all_are_leaves);

	if (mark_parent)
	    ret.marked_for_update_parent_msatol = mds;
	else
	    ret.marked_for_update_parent_msatol = null;
	
	ret.generative_support = MeasuredSupport.copyOf(mds.generative_support);
	ret.split_top_down_boosting_statistics = null;

	ret.tree_nodes_support = new Vector<>();
	int i;
	for (i=0;i<mds.tree_nodes_support.size();i++)
	    ret.tree_nodes_support.addElement(mds.tree_nodes_support.elementAt(i));

	return ret;
    }

    public static MeasuredSupportAtTupleOfNodes[] SPLIT_NODE(Node node, FeatureTest ft, int feature_split_index){
	// split the support and local examples (hard) of the node

	Feature feature_in_measured_support = node.node_support.feature(feature_split_index);
	
	MeasuredSupportAtTupleOfNodes[] ret = new MeasuredSupportAtTupleOfNodes[2];

	LocalEmpiricalMeasure [] split_meas;
	Feature [] split_feat;

	FeatureTest f = FeatureTest.copyOf(ft, feature_in_measured_support);

	String tvb = f.check_trivial_branching(node.myTree.myGET.myDS, feature_in_measured_support, true);

	if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
	    ret[0] = new MeasuredSupportAtTupleOfNodes(node.node_support, node.observation_indexes_in_node); // left
	    ret[1] = null; //right
	}else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
	    ret[0] = null; //left
	    ret[1] = new MeasuredSupportAtTupleOfNodes(node.node_support, node.observation_indexes_in_node); //right
	}else{
	    ret[0] = new MeasuredSupportAtTupleOfNodes(node.node_support, node.observation_indexes_in_node);
	    ret[1] = new MeasuredSupportAtTupleOfNodes(node.node_support, node.observation_indexes_in_node); 

	    split_meas = f.split_measure_hard(ret[0].generative_support.local_empirical_measure, node.myTree.myGET.myDS, feature_in_measured_support, true);
	    split_feat = f.split_feature(node.myTree.myGET.myDS, feature_in_measured_support, false, false);

	    ret[0].generative_support.local_empirical_measure = split_meas[0];
	    ret[0].generative_support.support.setFeatureAt(split_feat[0], feature_split_index);

	    ret[1].generative_support.local_empirical_measure = split_meas[1];
	    ret[1].generative_support.support.setFeatureAt(split_feat[1], feature_split_index);
	}
	    
	return ret;
    }

    public double p_R_independence(){
	double p_R = 1.0;
	int l;
	for (l=0;l<tree_nodes_support.size();l++)
	    p_R *= tree_nodes_support.elementAt(l).p_reach;
	return p_R;
    }
    
    public void check_all_tree_nodes_support_are_leaves(){
	int i;
    for (i = 0; i < tree_nodes_support.size(); i++)
      if (!tree_nodes_support.elementAt(i).is_leaf)
        Dataset.perror(
            "MeasuredSupportAtTupleOfNodes.class :: not all leaves ("
                + tree_nodes_support.elementAt(i)
                + ")");
    }

    public void unmark_for_update(Node new_left_node, Node new_right_node){
    if ((!marked_for_update)
        || (marked_for_update_index_tree == -1)
        || (marked_for_update_which_child.equals(MeasuredSupportAtTupleOfNodes.NO_CHILD)))
      Dataset.perror(
          "MeasuredSupportAtTupleOfNodes.class :: unmarking an unmarked"
              + " MeasuredSupportAtTupleOfNodes");

    if (marked_for_update_which_child.equals(MeasuredSupportAtTupleOfNodes.LEFT_CHILD))
      tree_nodes_support.setElementAt(new_left_node, marked_for_update_index_tree);
    else if (marked_for_update_which_child.equals(MeasuredSupportAtTupleOfNodes.RIGHT_CHILD))
      tree_nodes_support.setElementAt(new_right_node, marked_for_update_index_tree);
    else
      Dataset.perror(
          "MeasuredSupportAtTupleOfNodes.class :: no such token as "
              + marked_for_update_which_child);

	marked_for_update_parent_msatol = null;
	marked_for_update = false;
	marked_for_update_index_tree = -1;
	marked_for_update_which_child = MeasuredSupportAtTupleOfNodes.NO_CHILD;
    }
    
    public boolean tree_nodes_support_contains_node(Node n){
	if (tree_nodes_support.elementAt(n.myTree.name).equals(n))
	    return true;
	return false;
    }

    public void squash_for_missing_data_imputation(Observation o){
	// reduces to null all features that are NOT unknown in o
	int i;
	for (i=0;i<generative_support.support.dim();i++)
	    if (!Observation.FEATURE_IS_UNKNOWN(o, i))
		generative_support.support.setNullFeatureAt(i, geot.myDS);
    }
    
    public String toString(){
	String ret = "";
	ret = generative_support + "{{" + tree_nodes_support.size() + "}}";
	return ret;
    }
    
    public boolean done_for_missing_data_imputation(){
	return (tree_nodes_support.size()==0);
    }
    
    public void prepare_for_missing_data_imputation(){
	int i;
	for (i=tree_nodes_support.size()-1;i>=0;i--)
	    if (tree_nodes_support.elementAt(i).is_leaf)
		tree_nodes_support.removeElementAt(i);
    }

    public boolean done_for_all_supports(){
	return (tree_nodes_support.size()==0);
    }

    public void compute_split_top_down_boosting_statistics(FeatureTest [][] all_feature_tests, int feature_index, int feature_test_index){
	// MAKE SURE FEATURE i is splittable before calling this method
	split_top_down_boosting_statistics = split_for_boosting_computations(all_feature_tests[feature_index][feature_test_index], feature_index);
    }

    public double [] rapid_split_statistics(FeatureTest ft, int feature_split_index){
	// returns as double left_card, double right_card, double left_wud, double right_wud
	double [] ret = new double[4];
	double [] cd;
	Feature feature_in_measured_support = generative_support.support.feature(feature_split_index);

	FeatureTest f = FeatureTest.copyOf(ft, feature_in_measured_support);
	String tvb = f.check_trivial_branching(geot.myDS, feature_in_measured_support, true);
	double rat0;
	Feature [] split_feat;

	if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
	    ret[0] = (double) generative_support.local_empirical_measure.observations_indexes.length;
	    ret[1] = 0.0;
	    ret[2] = (double) generative_support.support.weight_uniform_distribution;
	    ret[3] = 0.0;
	}else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
	    ret[0] = 0.0;
	    ret[1] = (double) generative_support.local_empirical_measure.observations_indexes.length;
	    ret[2] = 0.0;
	    ret[3] = (double) generative_support.support.weight_uniform_distribution;
	}else{
	    cd = f.rapid_stats_split_measure_hard(generative_support.local_empirical_measure, geot.myDS, feature_in_measured_support, true);
	    ret[0] = cd[0];
	    ret[1] = cd[1];
	    
	    split_feat = f.split_feature(geot.myDS, feature_in_measured_support, false, false);
	    rat0 = split_feat[0].length() / generative_support.support.feature(feature_split_index).length();
	    ret[2] = generative_support.support.weight_uniform_distribution * rat0;
	    ret[3] = generative_support.support.weight_uniform_distribution * (1.0 - rat0);
	}

	return ret;
    }
    

    public MeasuredSupportAtTupleOfNodes[] split_for_boosting_computations(FeatureTest ft, int feature_split_index){
	// applies ft on generative_support.support
	// returns the left and right MeasuredSupportAtTupleOfNodes WITH NODE UNCHANGED (has to be changed by left / right child afterwards WHEN FEATURE CHOSEN IN BOOSTING)
	// completes otherwise everything: new supports, examples at the supports
	// IF one MeasuredSupportAtTupleOfNodes has no example reaching it, replaces it by null (simplifies boosting related computations)

	Feature feature_in_measured_support = generative_support.support.feature(feature_split_index);
	
	MeasuredSupportAtTupleOfNodes[] ret = new MeasuredSupportAtTupleOfNodes[2];

	LocalEmpiricalMeasure [] split_meas;
	Feature [] split_feat;

	FeatureTest f = FeatureTest.copyOf(ft, feature_in_measured_support);

	String tvb = f.check_trivial_branching(geot.myDS, feature_in_measured_support, true);
	
	if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
	    ret[0] = MeasuredSupportAtTupleOfNodes.copyOf(this, false, false, true); //left
	    ret[0].check_all_tree_nodes_support_are_leaves();
		
	    ret[1] = null; //right
	}else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
	    ret[0] = null; //left
	    
	    ret[1] = MeasuredSupportAtTupleOfNodes.copyOf(this, false, false, true); //right
	    ret[1].check_all_tree_nodes_support_are_leaves();
	}else{
	    ret[0] = MeasuredSupportAtTupleOfNodes.copyOf(this, false, false, true); //left
	    ret[0].check_all_tree_nodes_support_are_leaves();

	    ret[1] = MeasuredSupportAtTupleOfNodes.copyOf(this, false, false, true); //right
	    ret[1].check_all_tree_nodes_support_are_leaves();

	    split_meas = f.split_measure_hard(generative_support.local_empirical_measure, geot.myDS, feature_in_measured_support, true);
	    split_feat = f.split_feature(geot.myDS, feature_in_measured_support, false, false);

	    ret[0].generative_support.local_empirical_measure = split_meas[0];
	    ret[0].generative_support.support.setFeatureAt(split_feat[0], feature_split_index);

	    ret[1].generative_support.local_empirical_measure = split_meas[1];
	    ret[1].generative_support.support.setFeatureAt(split_feat[1], feature_split_index);
	}
	    
	return ret;
    }
    
    public MeasuredSupportAtTupleOfNodes[] split_for_all_supports(){
    // picks an element in tree_nodes_support (iteratively)
    // if the split is trivial (all the feature domain at left or right),
    // (i) updates tree_nodes_support with the relevant child and
    // (ii) return null
    // otherwise (trivial split)
    // (i) duplicate the MeasuredSupportAtTupleOfNodes in two using the split,
    // (ii) computes the two generative_support (support BUT NOT observations_indexes_in_support)
    // for the two children using the split
    // (iv) updates tree_nodes_support for the two new MDS and
    // (v) return an array [2] with the MDS'es

    if (done_for_missing_data_imputation())
      Dataset.perror("MeasuredSupportAtTupleOfNodes.class :: cannot try splitting: MDS finished");

	Node tree_node = tree_nodes_support.elementAt(0);
	int index_feature_split = tree_node.node_feature_split_index;

	Feature feature_in_tree_node =  tree_node.node_support.feature(index_feature_split);
	Feature feature_in_measured_support = generative_support.support.feature(index_feature_split);

	MeasuredSupportAtTupleOfNodes[] ret;
	LocalEmpiricalMeasure [] split_meas;
	Feature [] split_feat;
	
	FeatureTest f = FeatureTest.copyOf(tree_node.node_feature_test, feature_in_tree_node);
	String tvb = f.check_trivial_branching(geot.myDS, feature_in_measured_support, true);

	if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
	    tree_nodes_support.setElementAt(tree_node.left_child, 0);
	    // no change in observations_indexes_in_support
	    // no change in support
	    
	    if (tree_nodes_support.elementAt(0).is_leaf)
		tree_nodes_support.removeElementAt(0);
	    
	    ret = null;
	}else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
	    tree_nodes_support.setElementAt(tree_node.right_child, 0);
	    // no change in observations_indexes_in_support
	    // no change in support
	    
	    if (tree_nodes_support.elementAt(0).is_leaf)
		tree_nodes_support.removeElementAt(0);
	    
	    ret = null;
	}else{
	    ret = new MeasuredSupportAtTupleOfNodes[2]; 
	    ret[0] = MeasuredSupportAtTupleOfNodes.copyOf(this, true, false, false); //left
	    ret[1] = MeasuredSupportAtTupleOfNodes.copyOf(this, true, false, false); //right
	    
	    split_meas = f.split_measure_hard(generative_support.local_empirical_measure, geot.myDS, feature_in_measured_support, true); 
	    split_feat = f.split_feature(geot.myDS, feature_in_measured_support, false, false);

	    // left
	    if (tree_node.left_child.is_leaf)
		ret[0].tree_nodes_support.removeElementAt(0);
	    else
		ret[0].tree_nodes_support.setElementAt(tree_node.left_child, 0);
	    ret[0].generative_support.local_empirical_measure = split_meas[0];
	    ret[0].generative_support.support.setFeatureAt(split_feat[0], index_feature_split);

	    // right
	    if (tree_node.right_child.is_leaf)
		ret[1].tree_nodes_support.removeElementAt(0);
	    else
		ret[1].tree_nodes_support.setElementAt(tree_node.right_child, 0);
	    ret[1].generative_support.local_empirical_measure = split_meas[1];
	    ret[1].generative_support.support.setFeatureAt(split_feat[1], index_feature_split);
	}
	return ret;
    }

    public MeasuredSupportAtTupleOfNodes[] split_for_missing_data_imputation_generative_forest(Observation reference){
    // picks an element in tree_nodes_support (iteratively)
    // if the split is on a known feature of reference,
    // (i) updates tree_nodes_support with the relevant child
    // (ii) updates generative_support (observations_indexes_in_support) using the split and
    // (iii) return null
    // otherwise, if the split is trivial (all the feature domain at left or right),
    // (i) updates tree_nodes_support with the relevant child and
    // (ii) return null
    // otherwise (unknown feature AND non trivial split)
    // (i) duplicate the MeasuredSupportAtTupleOfNodes in two using the split,
    // (ii) computes the two generative_support (observations_indexes_in_support AND support) for
    // the two children using the split
    // (iv) updates tree_nodes_support for the two new MDS and
    // (v) return an array [2] with the MDS'es

    if (done_for_missing_data_imputation())
      Dataset.perror("MeasuredSupportAtTupleOfNodes.class :: cannot try splitting: MDS finished");

    if (!generative_support.local_empirical_measure.contains_indexes())
      Dataset.perror(
          "MeasuredSupportAtTupleOfNodes.class :: observations_indexes in support is null");

	Node tree_node = tree_nodes_support.elementAt(0);
	int index_feature_split = tree_node.node_feature_split_index;

	Feature feature_in_tree_node =  tree_node.node_support.feature(index_feature_split);
	Feature feature_in_measured_support = generative_support.support.feature(index_feature_split);
	
	MeasuredSupportAtTupleOfNodes[] ret;
	//int [][] split_obs;
	LocalEmpiricalMeasure [] split_meas;
	
	Feature [] split_feat;
	boolean goes_left, zero_measure;
	
	FeatureTest f = FeatureTest.copyOf(tree_node.node_feature_test, feature_in_tree_node);
	
	if (!Observation.FEATURE_IS_UNKNOWN(reference, index_feature_split)){
	    split_meas = f.split_measure_soft(generative_support.local_empirical_measure, geot.myDS, feature_in_tree_node);
	    goes_left = tree_node.node_feature_test.observation_goes_left(reference, geot.myDS, feature_in_tree_node, true);
	    zero_measure = false;

	    if ( ( (goes_left) && ( (split_meas[0] == null) || (split_meas[0].observations_indexes == null) ) ) ||
		 ( (!goes_left) && ( (split_meas[1] == null) || (split_meas[1].observations_indexes == null) ) ) )
		zero_measure = true; // happens if branches to a support without observation => we stop splitting support and keep it as is

	    if (!zero_measure){
		if (goes_left){
		    tree_nodes_support.setElementAt(tree_node.left_child, 0);
		    generative_support.local_empirical_measure = split_meas[0];
		}else{
		    tree_nodes_support.setElementAt(tree_node.right_child, 0);
		    generative_support.local_empirical_measure = split_meas[1];
		}
		if (tree_nodes_support.elementAt(0).is_leaf)
		    tree_nodes_support.removeElementAt(0);
	    }else{
		tree_nodes_support.removeAllElements(); // stops the splits
	    }
	    
	    ret = null;
	}else{
	    String tvb = f.check_trivial_branching(geot.myDS, feature_in_measured_support, true);
	    
	    if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
		tree_nodes_support.setElementAt(tree_node.left_child, 0);
		// no change in observations_indexes_in_support
		// no change in support

		if (tree_nodes_support.elementAt(0).is_leaf)
		    tree_nodes_support.removeElementAt(0);
		
		ret = null;
	    }else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
		tree_nodes_support.setElementAt(tree_node.right_child, 0);
		// no change in observations_indexes_in_support
		// no change in support

		if (tree_nodes_support.elementAt(0).is_leaf)
		    tree_nodes_support.removeElementAt(0);
		
		ret = null;
	    }else{		
		ret = new MeasuredSupportAtTupleOfNodes[2];
		ret[0] = MeasuredSupportAtTupleOfNodes.copyOf(this, true, false, false); //left
		ret[1] = MeasuredSupportAtTupleOfNodes.copyOf(this, true, false, false); //right

		split_meas = f.split_measure_soft(generative_support.local_empirical_measure, geot.myDS, feature_in_measured_support); 
		split_feat = f.split_feature(geot.myDS, feature_in_measured_support, false, false);

		if (split_meas[0] != null){
		    if (tree_node.left_child.is_leaf)
			ret[0].tree_nodes_support.removeElementAt(0);
		    else
			ret[0].tree_nodes_support.setElementAt(tree_node.left_child, 0);
		    ret[0].generative_support.local_empirical_measure = split_meas[0];

		    ret[0].generative_support.support.setFeatureAt(split_feat[0], index_feature_split);
		}else
		    ret[0] = null;

		// right
		if (split_meas[1] != null){
		    if (tree_node.right_child.is_leaf)
			ret[1].tree_nodes_support.removeElementAt(0);
		    else
			ret[1].tree_nodes_support.setElementAt(tree_node.right_child, 0);
		    ret[1].generative_support.local_empirical_measure = split_meas[1];
		    ret[1].generative_support.support.setFeatureAt(split_feat[1], index_feature_split);
		}else
		    ret[1] = null;
	    }
	}
	
	return ret;
    }

    
    public MeasuredSupportAtTupleOfNodes[] split_for_all_supports_with_positive_measure(){
    // picks an element in tree_nodes_support (iteratively)
    // if the split is trivial (all the feature domain at left or right),
    // (i) updates tree_nodes_support with the relevant child and
    // (ii) return null
    // otherwise (trivial split)
    // (i) duplicate the MeasuredSupportAtTupleOfNodes in two using the split,
    // (ii) computes the two generative_support (support BUT NOT observations_indexes_in_support)
    // for the two children using the split
    // (iv) updates tree_nodes_support for the two new MDS and
    // (v) return an array [2] with the MDS'es

    if (done_for_all_supports())
      Dataset.perror("MeasuredSupportAtTupleOfNodes.class :: cannot try splitting: MDS finished");

	Node tree_node = tree_nodes_support.elementAt(0);
	int index_feature_split = tree_node.node_feature_split_index;

	Feature feature_in_tree_node =  tree_node.node_support.feature(index_feature_split);
	Feature feature_in_measured_support = generative_support.support.feature(index_feature_split);

	MeasuredSupportAtTupleOfNodes[] ret;
	LocalEmpiricalMeasure [] split_meas;
	Feature [] split_feat;
	
	FeatureTest f = FeatureTest.copyOf(tree_node.node_feature_test, feature_in_tree_node);
	String tvb = f.check_trivial_branching(geot.myDS, feature_in_measured_support, true);

	if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
	    tree_nodes_support.setElementAt(tree_node.left_child, 0);
	    // no change in observations_indexes_in_support
	    // no change in support
	    
	    if (tree_nodes_support.elementAt(0).is_leaf)
		tree_nodes_support.removeElementAt(0);
	    
	    ret = null;
	}else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
	    tree_nodes_support.setElementAt(tree_node.right_child, 0);
	    // no change in observations_indexes_in_support
	    // no change in support
	    
	    if (tree_nodes_support.elementAt(0).is_leaf)
		tree_nodes_support.removeElementAt(0);
	    
	    ret = null;
	}else{
	    ret = new MeasuredSupportAtTupleOfNodes[2]; 
	    ret[0] = MeasuredSupportAtTupleOfNodes.copyOf(this, true, false, false); //left
	    ret[1] = MeasuredSupportAtTupleOfNodes.copyOf(this, true, false, false); //right
	    
	    split_meas = f.split_measure_soft(generative_support.local_empirical_measure, geot.myDS, feature_in_measured_support);
	    split_feat = f.split_feature(geot.myDS, feature_in_measured_support, false, false);

	    // left
	    if (split_meas[0] != null){
		if (tree_node.left_child.is_leaf)
		    ret[0].tree_nodes_support.removeElementAt(0);
		else
		    ret[0].tree_nodes_support.setElementAt(tree_node.left_child, 0);
		ret[0].generative_support.local_empirical_measure = split_meas[0];
		ret[0].generative_support.support.setFeatureAt(split_feat[0], index_feature_split);
	    }else{
		ret[0] = null;
	    }
	    
	    // right
	    if (split_meas[1] != null){
		if (tree_node.right_child.is_leaf)
		    ret[1].tree_nodes_support.removeElementAt(0);
		else
		    ret[1].tree_nodes_support.setElementAt(tree_node.right_child, 0);
		ret[1].generative_support.local_empirical_measure = split_meas[1];
		ret[1].generative_support.support.setFeatureAt(split_feat[1], index_feature_split);
	    }else{
		ret[1] = null;
	    }
	}
	return ret;
    }
    
    public MeasuredSupportAtTupleOfNodes[] split_to_find_all_supports_of_reference_in_generative_forest(Observation reference){

    if (!generative_support.local_empirical_measure.contains_indexes())
      Dataset.perror(
          "MeasuredSupportAtTupleOfNodes.class :: observations_indexes in support is null");

	Node tree_node = tree_nodes_support.elementAt(0);
	int index_feature_split = tree_node.node_feature_split_index;

	Feature feature_in_tree_node =  tree_node.node_support.feature(index_feature_split);
	Feature feature_in_measured_support = generative_support.support.feature(index_feature_split);
	
	MeasuredSupportAtTupleOfNodes[] ret;
	LocalEmpiricalMeasure [] split_meas;
	
	Feature [] split_feat;
	boolean goes_left, zero_measure;
	
	FeatureTest f = FeatureTest.copyOf(tree_node.node_feature_test, feature_in_tree_node);
	
	if (!Observation.FEATURE_IS_UNKNOWN(reference, index_feature_split)){
      if (!feature_in_measured_support.has_in_range(
          reference.typed_features.elementAt(index_feature_split)))
        Dataset.perror(
            "MeasuredSupportAtTupleOfNodes.class :: "
                + feature_in_measured_support
                + " does not contain value "
                + reference.typed_features.elementAt(index_feature_split));

	    String tvb = f.check_trivial_branching(geot.myDS, feature_in_measured_support, true);

	    if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
		tree_nodes_support.setElementAt(tree_node.left_child, 0);
		// no change in observations_indexes_in_support
		// no change in support

		if (tree_nodes_support.elementAt(0).is_leaf)
		    tree_nodes_support.removeElementAt(0);
		
		ret = null;
	    }else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
		tree_nodes_support.setElementAt(tree_node.right_child, 0);
		// no change in observations_indexes_in_support
		// no change in support

		if (tree_nodes_support.elementAt(0).is_leaf)
		    tree_nodes_support.removeElementAt(0);
		
		ret = null;
	    }else{	    
		split_meas = f.split_measure_soft(generative_support.local_empirical_measure, geot.myDS, feature_in_measured_support); 
		split_feat = f.split_feature(geot.myDS, feature_in_measured_support, false, false);

		goes_left = tree_node.node_feature_test.observation_goes_left(reference, geot.myDS, feature_in_tree_node, true);
		zero_measure = false;

		if ( ( (goes_left) && ( (split_meas[0] == null) || (split_meas[0].observations_indexes == null) ) ) ||
		     ( (!goes_left) && ( (split_meas[1] == null) || (split_meas[1].observations_indexes == null) ) ) )
		    zero_measure = true; // happens if branches to a support without observation => we stop splitting support and keep it as is

		if (!zero_measure){
		    if (goes_left){
			tree_nodes_support.setElementAt(tree_node.left_child, 0);
			generative_support.local_empirical_measure = split_meas[0];
			generative_support.support.setFeatureAt(split_feat[0], index_feature_split);
			
		    }else{
			tree_nodes_support.setElementAt(tree_node.right_child, 0);
			generative_support.local_empirical_measure = split_meas[1];
			generative_support.support.setFeatureAt(split_feat[1], index_feature_split);
		    }
		    if (tree_nodes_support.elementAt(0).is_leaf)
			tree_nodes_support.removeElementAt(0);
		}else{
		    tree_nodes_support.removeAllElements(); // stops the splits
		}
	    }
	    
	    ret = null;
	}else{
	    String tvb = f.check_trivial_branching(geot.myDS, feature_in_measured_support, true);
	    
	    if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
		tree_nodes_support.setElementAt(tree_node.left_child, 0);
		// no change in observations_indexes_in_support
		// no change in support

		if (tree_nodes_support.elementAt(0).is_leaf)
		    tree_nodes_support.removeElementAt(0);
		
		ret = null;
	    }else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
		tree_nodes_support.setElementAt(tree_node.right_child, 0);
		// no change in observations_indexes_in_support
		// no change in support

		if (tree_nodes_support.elementAt(0).is_leaf)
		    tree_nodes_support.removeElementAt(0);
		
		ret = null;
	    }else{		
		ret = new MeasuredSupportAtTupleOfNodes[2];
		ret[0] = MeasuredSupportAtTupleOfNodes.copyOf(this, true, false, false); //left
		ret[1] = MeasuredSupportAtTupleOfNodes.copyOf(this, true, false, false); //right

		split_meas = f.split_measure_soft(generative_support.local_empirical_measure, geot.myDS, feature_in_measured_support); 
		split_feat = f.split_feature(geot.myDS, feature_in_measured_support, false, false);

		if (split_meas[0] != null){
		    if (tree_node.left_child.is_leaf)
			ret[0].tree_nodes_support.removeElementAt(0);
		    else
			ret[0].tree_nodes_support.setElementAt(tree_node.left_child, 0);
		    ret[0].generative_support.local_empirical_measure = split_meas[0];

		    ret[0].generative_support.support.setFeatureAt(split_feat[0], index_feature_split);
		}else
		    ret[0] = null;

		// right
		if (split_meas[1] != null){
		    if (tree_node.right_child.is_leaf)
			ret[1].tree_nodes_support.removeElementAt(0);
		    else
			ret[1].tree_nodes_support.setElementAt(tree_node.right_child, 0);
		    ret[1].generative_support.local_empirical_measure = split_meas[1];
		    ret[1].generative_support.support.setFeatureAt(split_feat[1], index_feature_split);
		}else
		    ret[1] = null;
	    }
	}
	
	return ret;
    }
}

class MeasuredSupport implements Debuggable{
    // used to generate data with GenerativeModelBasedOnEnsembleOfTrees models
    // for GEOT
    
    public Support support;
    public LocalEmpiricalMeasure local_empirical_measure;
    
    GenerativeModelBasedOnEnsembleOfTrees myGET;

    MeasuredSupport(GenerativeModelBasedOnEnsembleOfTrees geot, Support s){
	myGET = geot;
	support = s;

	local_empirical_measure = null;
    }

    MeasuredSupport(Support s, int [] observations_indexes){
	myGET = null;
	support = Support.copyOf(s);
	local_empirical_measure = new LocalEmpiricalMeasure(observations_indexes.length);
	local_empirical_measure.init_indexes(observations_indexes);
    }
    
    MeasuredSupport(GenerativeModelBasedOnEnsembleOfTrees geot, int exception){
	this(geot, geot.myDS.domain_support());
	int [] all_indexes = geot.myDS.all_indexes(exception);
	local_empirical_measure = new LocalEmpiricalMeasure(all_indexes.length);
	local_empirical_measure.init_indexes(all_indexes);
    }

    public static MeasuredSupport copyOf(MeasuredSupport gs){
	MeasuredSupport g = new MeasuredSupport(gs.myGET, Support.copyOf(gs.support));
	g.local_empirical_measure = new LocalEmpiricalMeasure(gs.local_empirical_measure.observations_indexes.length);
	g.local_empirical_measure.init_indexes(gs.local_empirical_measure.observations_indexes);
	return g;
    }

    public String toString(){
	String ret = "";
	ret += support;
	ret += "{" + local_empirical_measure.total_number_of_indexes() + "}";
	return ret;
    }

}
