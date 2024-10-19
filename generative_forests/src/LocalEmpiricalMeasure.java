// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class LocalEmpiricalMeasure (convenience)
 *****/

class LocalEmpiricalMeasure implements Debuggable{
    // Use observations_indexes, proportions for GEOT, total_weight for EOGT
    
    public int [] observations_indexes;
    public double [] proportions;

    public double total_weight;

    LocalEmpiricalMeasure(int n){

	if (n == 0){
	    observations_indexes = null;
	    proportions = null;
	    total_weight = 0.0;
	}else{
	    observations_indexes = new int[n];
	    proportions = new double[n];
	    
	    int i;
	    for (i=0;i<n;i++){
		observations_indexes[i] = -1;
		proportions[i] = -1.0;
	    }
	    total_weight = 0.0;
	}
    }

    public static LocalEmpiricalMeasure[] SOFT_SPLIT_AT_NODE(Node node, LocalEmpiricalMeasure lem_at_n){ 

	LocalEmpiricalMeasure[] ret;

	if (lem_at_n == null){
	    ret = new LocalEmpiricalMeasure[2];
	    ret[0] = ret[1] = null;
	    return ret;
	}
	
	int index_feature_split = node.node_feature_split_index;
	Feature feature_in_node =  node.node_support.feature(index_feature_split);	
	
	FeatureTest f = FeatureTest.copyOf(node.node_feature_test, feature_in_node);

	ret = f.split_measure_soft(lem_at_n, node.myTree.myGET.myDS, feature_in_node); 

	return ret;
    }

    public String toString(){
	if (observations_indexes == null)
	    return null;
	int i;
	String ret = "{";
	for (i=0;i<observations_indexes.length;i++){
	    ret += "(" + observations_indexes[i] + ", " + proportions[i] + ")";
	}
	ret += "}";
	return ret;
    }
    
    public boolean contains_indexes(){
	if ( (observations_indexes == null) || (observations_indexes.length == 0) )
	    return false;
	return true;
    }

    public void add(int iv, double pv){
	// checks the index is not yet in observations_indexes
	if (contains_index(iv))
	    Dataset.perror("LocalEmpiricalMeasure.class :: cannot add index");

	int [] new_indexes;
	double [] new_proportions;
	int former_size;
	
	if (observations_indexes == null){
	    former_size = 0;
	}else
	    former_size = observations_indexes.length;

	new_indexes = new int[former_size + 1];
	new_proportions = new double[former_size + 1];

	int i;
	for (i=0;i<former_size;i++){
	    new_indexes[i] = observations_indexes[i];
	    new_proportions[i] = proportions[i];
	}
	new_indexes[former_size] = iv;
	new_proportions[former_size] = pv;

	observations_indexes = new_indexes;
	proportions = new_proportions;

	total_weight += pv;
    }
    
    public boolean contains_index(int v){
	if (observations_indexes == null)
	    return false;
	int i=0;
	for (i=0;i<observations_indexes.length;i++)
	    if (observations_indexes[i] == v)
		return true;
	return false;
    }

    public void init_indexes(int [] v){
	observations_indexes = new int[v.length];
	proportions = new double[v.length];
	int i;
	for (i=0;i<v.length;i++){
	    observations_indexes[i] = v[i];
	    proportions[i] = -1.0;
	}
    }

    public void init_proportions(double v){
	if (observations_indexes == null)
	    Dataset.perror("LocalEmpiricalMeasure.class :: observations_indexes == null");

    if (proportions.length != observations_indexes.length)
      Dataset.perror(
          "LocalEmpiricalMeasure.class :: proportions.length != observations_indexes.length");

	int i;
	total_weight = 0.0;
	for (i=0;i<proportions.length;i++){
	    proportions[i] = v;
	    total_weight += v;
	}
    }

    public int total_number_of_indexes(){
	return observations_indexes.length;
    }

    public double total_weight(){
	if (observations_indexes == null)
	    return 0.0;
	double ret = 0.0;
	int i;
	for (i=0;i<proportions.length;i++)
	    if (proportions[i] <= 0.0)
		Dataset.perror("LocalEmpiricalFeatures.class :: non >0 proportion");
	    else
		ret += proportions[i];
	return ret;
    }


}

