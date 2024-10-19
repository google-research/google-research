// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Utils 
 *****/

class Utils implements Debuggable{
    int domain_id;
    public static Random r = new Random();

    public static boolean RANDOMISE_SPLIT_FINDING_WHEN_TOO_MANY_SPLITS = true;

    public static int MAX_CARD_MODALITIES = 200;
    public static int MAX_CARD_MODALITIES_BEFORE_RANDOMISATION = 10;
    // for NOMINAL variables: partially computes the number of candidates splits if modalities > MAX_CARD_MODALITIES_BEFORE_RANDOMISATION
    // with 22, Mem <= 3 Mb (tests)
    
    public static int MAX_SIZE_FOR_RANDOMISATION = 2; // used to partially compute the number of candidates splits for NOMINAL variables (here, O({n\choose MAX_SIZE_FOR_RANDOMISATION}))

    public static int GUESS_MAX_TRUE_IN_BOOLARRAY(int bitset_size){
    if (bitset_size > MAX_CARD_MODALITIES)
      Dataset.perror(
          "Utils.class :: "
              + bitset_size
              + "modalities, too large (MAX = "
              + MAX_CARD_MODALITIES
              + ".");

	if (bitset_size <= Utils.MAX_CARD_MODALITIES_BEFORE_RANDOMISATION)
	    return bitset_size;

	int v = (int) ((double) Utils.MAX_CARD_MODALITIES_BEFORE_RANDOMISATION / (Math.log(Utils.MAX_CARD_MODALITIES_BEFORE_RANDOMISATION) + 1.0));
	if (bitset_size >= 50)
	    v--;
	if (bitset_size >= 100)
	    v--;

	return v;
    }

    public static int [] NEXT_INDEXES(int [] current_indexes, int [] max_indexes, int forbidden_index){
	if (current_indexes.length != max_indexes.length)
	    Dataset.perror("Utils.class :: not the same length");
	int index_increase = max_indexes.length-1, i;
	boolean found = false;
	do{
	    if ( (current_indexes[index_increase] < max_indexes[index_increase]) &&
		 ( (forbidden_index == -1) || ( (forbidden_index != -1) && (index_increase != forbidden_index) ) ) )
		found = true;
	    else 
		index_increase--;
	}while( (index_increase >= 0) && (!found) );

	if (!found)
	    return null;
	
	int [] nextint = new int [current_indexes.length];
	System.arraycopy(current_indexes, 0, nextint, 0, current_indexes.length);
	nextint[index_increase]++;
	for (i=index_increase+1;i<current_indexes.length;i++)
	    if ( (forbidden_index == -1) || ( (forbidden_index != -1) && (i != forbidden_index) ) )
		nextint[i] = 0;
	return nextint;
    }

    public static Vector<Vector<Node>> TUPLES_OF_NODES_WITH_NON_EMPTY_SUPPORT_INTERSECTION(Vector<Tree> all_trees, Node node, boolean ensure_non_zero_measure){
	int [] current_indexes = new int [all_trees.size()];
	int [] max_indexes = new int [all_trees.size()];
	int forbidden_index = -1, node_index;

	int i;
	//compute_temporary_leaves
	for (i=0;i<all_trees.size();i++)
	    all_trees.elementAt(i).compute_temporary_leaves();

	if (node != null){
	    // look for node_index in its tree's leaves
	    node_index = all_trees.elementAt(node.myTree.name).temporary_leaves.indexOf(node);
	    if (node_index == -1)
		Dataset.perror("Utils.class :: leaf not found");
	    
	    current_indexes[node.myTree.name] = node_index;
	    forbidden_index = node.myTree.name;
	}

	Vector <Node> new_element;
	Vector <Vector<Node>> set = new Vector<>();

	long max = 1, iter = 0;
	for (i=0;i<all_trees.size();i++){
	    max_indexes[i] = all_trees.elementAt(i).temporary_leaves.size() - 1;
	    max *= all_trees.elementAt(i).temporary_leaves.size();
	}

	do{
	    new_element = new Vector<>();
	    for (i=0;i<current_indexes.length;i++){
		new_element.addElement(all_trees.elementAt(i).temporary_leaves.elementAt(current_indexes[i]));
	    }
	    if (!Support.SUPPORT_INTERSECTION_IS_EMPTY(new_element, ensure_non_zero_measure, node.myTree.myGET.myDS))
		set.addElement(new_element);
	    current_indexes = NEXT_INDEXES(current_indexes, max_indexes,  forbidden_index);
	    iter++;

	    if (iter%(max/10) == 0)
		System.out.print((iter/(max/10))*10 + "% ");
	    
	}while(current_indexes != null);

	//discard_temporary_leaves
	for (i=0;i<all_trees.size();i++)
	    all_trees.elementAt(i).discard_temporary_leaves();
	
	return set;
    }

    public static Vector <BoolArray> ALL_NON_TRIVIAL_BOOLARRAYS(int bitset_size, int max_true_in_bitset){
	if (max_true_in_bitset > bitset_size)
	    Dataset.perror("Utils.class :: bad parameters for ALL_NON_TRIVIAL_BOOLARRAYS");
	
	Vector <BoolArray> ret = new Vector <>();
	MOVE_IN(ret, bitset_size, 0, max_true_in_bitset);
	ret.removeElementAt(0);
	if (ret.elementAt(ret.size()-1).cardinality() == bitset_size)
	    ret.removeElementAt(ret.size()-1);
	
	return ret;
    }

    public static void MOVE_IN(Vector<BoolArray> grow, int bitset_size, int index, int max_true_in_bitset){
	if (grow.size() == 0){
	    grow.addElement(new BoolArray(bitset_size));
	}
	BoolArray vv, v;
	int i, sinit = grow.size(), j;
	for (i=0;i<sinit;i++){
	    vv = grow.elementAt(i);
	    if (vv.cardinality() < max_true_in_bitset){
		v = (BoolArray) vv.duplicate();
		v.set(index, true);
		grow.addElement(v);
	    }
	}
	if (index < bitset_size - 1)
	    MOVE_IN(grow, bitset_size, index + 1, max_true_in_bitset);
    }    
    
}
