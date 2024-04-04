// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Tree
 *****/

class Tree implements Debuggable{
    int name, depth;
    
    Node root;
    // root of the tree

    Node star_node;
    // for data generation

    GenerativeModelBasedOnEnsembleOfTrees myGET;

    HashSet <Node> leaves;
    Vector <Node> temporary_leaves;
    // TEMPORARY convenience to map the leaves of the tree in a vector (gives access to an index);

    int number_nodes;
    //includes leaves

    int [] statistics_number_of_nodes_for_each_feature;

    Tree(GenerativeModelBasedOnEnsembleOfTrees eot, int v){
	myGET = eot;
	root = new Node(this, 0, 0, myGET.myDS.all_indexes(-1), myGET.myDS.domain_support());
	root.p_reach = 1.0;
	
	star_node = root;

	leaves = new HashSet <> ();
	leaves.add(root);
	
	name = v;
	depth = 0;
	number_nodes = 1;
	temporary_leaves = null;

	statistics_number_of_nodes_for_each_feature = new int[eot.myDS.features.size()];
    }

    public void compute_temporary_leaves(){
	temporary_leaves = new Vector<>();
	Iterator it = leaves.iterator();
	
	while(it.hasNext()){
	    temporary_leaves.addElement((Node) it.next());
	}
    }

    public void discard_temporary_leaves(){
	temporary_leaves = null;
    }
    
    public void checkConsistency(){
	int i, totn = 0;
	Node dumn;

	Iterator it = leaves.iterator();
	while(it.hasNext()){
	    dumn = (Node) it.next();
	    totn += (dumn.observations_in_node) ? dumn.observation_indexes_in_node.length : 0;
	}

    if (totn != myGET.myDS.observations_from_file.size())
      Dataset.perror(
          "Tree.class :: total number of examples reaching leaves != from the dataset size");
    }

    public void add_leaf_to_tree_structure(Node n){
	n.is_leaf = true;
	boolean ok;
	
	ok = leaves.add(n);
	if (!ok)
	    Dataset.perror("Tree.class :: adding a leaf already in leaves");
    }

    public boolean remove_leaf_from_tree_structure(Node n){
	if (!leaves.contains(n))
	    Dataset.perror("Tree.class :: Node " + n + " not in leaves ");

	boolean ok;

	ok = leaves.remove(n);
	if (!ok)
	    return false;

	return true;
    }

    public String toString(){
	int i;
    String v =
        "(name = #"
            + myGET.name
            + "."
            + name
            + " | depth = "
            + depth
            + " | #nodes = "
            + number_nodes
            + ")\n";
	Node dumn;
	
	v += root.display(new HashSet <Integer> (), myGET, -1.0);
	
        v += "Leaves:";

	Iterator it = leaves.iterator();
	while(it.hasNext()){
	    v += " ";
	    dumn = (Node) it.next();
	    v += "#" + dumn.name + dumn.observations_string(-1.0);
	}
	v += ".\n";
	
	return v;
    }

    // generation related stuff
    
    public void init_generation(){
	star_node = root;
    }

    public boolean generation_done(){
	return star_node.is_leaf;
    }

    // generation related stuff

    public MeasuredSupport update_star_node_and_support(MeasuredSupport gs){
	if (star_node.is_leaf)
	    return gs;

	if (!gs.support.is_subset_of(star_node.node_support))
	    Dataset.perror("Tree.class :: generative support not a subset of the star node's");

	if (gs.local_empirical_measure.observations_indexes.length == 0)
	    Dataset.perror("Tree.class :: current generative support has 0 empirical measure");
	
	MeasuredSupport generative_support_left = new MeasuredSupport(gs.myGET, star_node.left_child.node_support.cap(gs.support, false));
	MeasuredSupport generative_support_right = new MeasuredSupport(gs.myGET, star_node.right_child.node_support.cap(gs.support, false));

	int index_feature_split = star_node.node_feature_split_index;

	Feature feature_in_star_node =  star_node.node_support.feature(index_feature_split);
	Feature feature_in_measured_support = gs.support.feature(index_feature_split);
	
	FeatureTest ft = FeatureTest.copyOf(star_node.node_feature_test, feature_in_star_node);
	
	// check if branching trivial, does not seek non zero measure sets because there might be observations on [a,a] for generation
	String ttb = ft.check_trivial_branching(myGET.myDS, feature_in_measured_support, false);

	if (ttb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
	    generative_support_left.local_empirical_measure = gs.local_empirical_measure;
	    star_node = star_node.left_child;
	    
	    return generative_support_left;
	}else if (ttb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
	    generative_support_right.local_empirical_measure = gs.local_empirical_measure;
	    star_node = star_node.right_child;

	    return generative_support_right;
	}

	// no trivial branching
	// fast trick: use the star_node split to split gs.support and get the observed measure left / right

	LocalEmpiricalMeasure [] split_measure_at_support = ft.split_measure_soft(gs.local_empirical_measure, myGET.myDS, feature_in_measured_support);
	// Note: observations w/ missing values are subject to the (biased) random assignation again (we do not reuse the results of training's random assignations)
	// the supports obtained may thus be of variable empirical measure depending on missing values at generation time
	
	generative_support_left.local_empirical_measure = split_measure_at_support[0];
	generative_support_right.local_empirical_measure = split_measure_at_support[1];

	boolean pick_left;
	double p_left = -1.0, p_u_left, p_u_right, p_r_left, p_r_right, tw, gw;

	if (split_measure_at_support[1] == null)
	    pick_left = true;
	else if (split_measure_at_support[0] == null)
	    pick_left = false;
	else{
	    if (myGET.generative_forest){
		tw = (double) split_measure_at_support[0].total_weight;
		gw = (double) gs.local_empirical_measure.total_weight;

		p_left = tw / gw;
	    }else if (myGET.ensemble_of_generative_trees){
		// Using prop assumption, see draft

		p_r_left = star_node.left_child.p_reach * generative_support_left.support.volume / star_node.left_child.node_support.volume;
		p_r_right = star_node.right_child.p_reach * generative_support_right.support.volume / star_node.right_child.node_support.volume;

		p_left = p_r_left / (p_r_left + p_r_right);	
	    }else
		Dataset.perror("Tree.class :: cannot generate observations");
	    
	    if (Statistics.RANDOM_P_NOT(p_left) < p_left)
		pick_left = true;
	    else
		pick_left = false;
	}

	if (pick_left){
	    star_node = star_node.left_child;

	    return generative_support_left;
	}else{
	    star_node = star_node.right_child;

	    return generative_support_right;
	}
    }
}

