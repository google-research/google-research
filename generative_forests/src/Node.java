// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Node
 *****/

public class Node implements Debuggable{
    int name, depth, number_nodes;

    Node left_child, right_child;
    double p_left, p_right;
    // used iff myTree.myGET.ensemble_of_generative_trees == true

    double p_reach;
    // used iff myTree.myGET.ensemble_of_generative_trees == true
    
    boolean is_leaf;
    // if true, left_child = right_child = null;
    
    int [] observation_indexes_in_node;
    // index of observations that reach the node
    boolean observations_in_node;
        
    // Feature information at the node level
    Tree myTree;

    // Feature designing the split at the node
    
    FeatureTest node_feature_test;
    // feature test for the node
    
    int node_feature_split_index;
    // handle on the feature index in node_support
    
    Support node_support;
    
    Node(){
	name = -1;
	
	left_child = right_child = null;
	p_left = p_right = p_reach = -1.0;
	
	observation_indexes_in_node = null;
	is_leaf = true;
	observations_in_node = false;

	node_feature_split_index = -1;

	node_feature_test = null;
	node_support = null;
    }

    Node(Tree t, int v, int d){
	this();
	myTree = t;
	name = v;
	depth = d;
    }

    Node(Tree t, int v, int d, int [] observations_indexes, Support sn){
	this(t, v, d);
	
	if (sn == null)
	    Dataset.perror("Node.class :: no support given for new node");

	node_support = sn;

	if ( (observations_indexes == null) || (observations_indexes.length ==0) )
	    observations_in_node = false;
	else
	    observations_in_node = true;

	if (observations_in_node){
	    observation_indexes_in_node = new int [observations_indexes.length];
	    System.arraycopy(observations_indexes, 0, observation_indexes_in_node, 0, observations_indexes.length);
	}
    }

    public void recursive_compute_probabilities_soft(LocalEmpiricalMeasure lem){
	if (is_leaf)
	    return;

	LocalEmpiricalMeasure [] lem_split = LocalEmpiricalMeasure.SOFT_SPLIT_AT_NODE(this, lem);

	double p_tot = 0.0, p_l = 0.0, p_r = 0.0;

	int i;
	if (lem != null)
	    for (i=0;i<lem.proportions.length;i++){
		if (lem.proportions[i] == -1.0)
		    Dataset.perror("Node.class :: lem.proportions not initialized");
		p_tot += lem.proportions[i];
	    }

	p_tot /= (double) myTree.myGET.myDS.observations_from_file.size();

	if (lem_split[0] != null)
	    for (i=0;i<lem_split[0].proportions.length;i++){
		if (lem_split[0].proportions[i] == -1.0)
		    Dataset.perror("Node.class :: lem_split[0].proportions not initialized");
		p_l += lem_split[0].proportions[i];
	    }

	p_l /= (double) myTree.myGET.myDS.observations_from_file.size();
	
	if (lem_split[1] != null)
	    for (i=0;i<lem_split[1].proportions.length;i++){
		if (lem_split[1].proportions[i] == -1.0)
		    Dataset.perror("Node.class :: lem_split[1].proportions not initialized");
		p_r += lem_split[1].proportions[i];
	    }

	p_r /= (double) myTree.myGET.myDS.observations_from_file.size();

    if (!Statistics.APPROXIMATELY_EQUAL(p_l + p_r, p_tot, EPS2))
      Dataset.perror(
          "Node.class :: p mismatch -- "
              + p_l
              + " + "
              + p_r
              + " = "
              + (p_l + p_r)
              + " != "
              + p_tot);

	// just to ensure proper summation
	p_r = p_tot - p_l;
	
	if (lem == null){
	    p_reach = 0.0;
	    left_child.p_reach = 0.0;
	    right_child.p_reach = 0.0;

	    p_left = 0.5;
	    p_right = 0.5;
	}else{
	    if (p_tot == 0.0)
		Dataset.perror("Node.class :: p_tot == 0.0");
	    
	    p_left = p_l / p_tot;
	    p_right = p_r / p_tot;

	    left_child.p_reach = p_l;
	    right_child.p_reach = p_r;
	}
	
	left_child.recursive_compute_probabilities_soft(lem_split[0]);
	right_child.recursive_compute_probabilities_soft(lem_split[1]);
    }
    
    public boolean equals(Object o){
	if (o == this)
	    return true;
	if (!(o instanceof Node))
	    return false;
	Node test = (Node) o;
	if ( (test.name == name) && (test.myTree.name == myTree.name) && (test.depth == depth) )
	    return true;
	return false;
    }

    public String observations_string(double prob){
	if (prob < 0.0){
	    if (observations_in_node)
		return "{" + observation_indexes_in_node.length + "}";
	    else
		return "{}";
	}else
	    return "";
    }
    
    public String toString(double prob){
	String v = "";
	int leftn, rightn;

	if (prob >= 0.0)
	    v += "(" + DF4.format(prob) + ")";

	if (name != 0)
	    v += "[#" + name + "]";
	else
	    v += "[#0:root]";

	if (is_leaf) 
	    v += " leaf " + observations_string(prob); 
	else{
	    if (left_child != null)
		leftn = left_child.name;
	    else
		leftn = -1;
	    
	    if (right_child != null)
		rightn = right_child.name;
	    else
		rightn = -1;

      v +=
          " internal "
              + observations_string(prob)
              + " ("
              + node_feature_test.toString(myTree.myGET.myDS)
              + " ? #"
              + leftn
              + " : #"
              + rightn
              + ")";
	}

	v += "\n";

	return v;
    }

    public String toString(){
	String v = "";
	int leftn, rightn;

	if (name != 0)
	    v += "[#" + name + "]";
	else
	    v += "[#0:root]";

	if (is_leaf) 
	    v += " leaf " + observations_string(0.0); 
	else{
	    if (left_child != null)
		leftn = left_child.name;
	    else
		leftn = -1;
	    
	    if (right_child != null)
		rightn = right_child.name;
	    else
		rightn = -1;

      v +=
          " internal "
              + observations_string(0.0)
              + " ("
              + node_feature_test.toString(myTree.myGET.myDS)
              + " ? #"
              + leftn
              + " : #"
              + rightn
              + ")";
	}

	v += "\n";

	return v;
    }

    
    public String display(HashSet <Integer> indexes, GenerativeModelBasedOnEnsembleOfTrees eot, double prob){
	// eot, prob used iff EOGT
	
	String v = "", t;
	int i;
	HashSet <Integer> dum;
	boolean bdum;

	if (observations_in_node)
	    t = "\u2501";
	else
	    t = "~";
	    
	for (i=0;i<depth;i++){
	    if ( (i==depth-1) && (indexes.contains(new Integer(i))) )
		v += "\u2523" + t;
	    else if (i==depth-1)
		v += "\u2517" + t;
	    else if (indexes.contains(new Integer(i)))
		v += "\u2503 ";
	    else
		v += "  ";
	}
	if (eot.ensemble_of_generative_trees)
	    v += toString(prob);
	else
	    v += toString(-1.0);

	if (!is_leaf){
	    dum = new HashSet<Integer>(indexes);
	    bdum = dum.add(new Integer(depth));

	    if (left_child != null)
		v += left_child.display(dum, eot, p_left);
	    else
		v += "null";

	    if (right_child != null)
		v += right_child.display(indexes, eot, p_right);
	    else
		v += "null";
	}
	
	return v;
    }
    
    public void recursive_fill_node_counts(int [] node_counts){
	boolean found = false;
	int i = 0;
	
	if (!is_leaf){
	    node_counts[node_feature_split_index]++;
	    left_child.recursive_fill_node_counts(node_counts);
	    right_child.recursive_fill_node_counts(node_counts);
	}
    }

    public boolean has_feature_tests(){
	// returns true iff at least one feature can be split
	int i;
	
	for (i=0;i<node_support.dim();i++){
	    if (FeatureTest.HAS_AT_LEAST_ONE_FEATURE_TEST(node_support.feature(i), myTree.myGET.myDS))
		return true;
	}
	return false;
    }
}

