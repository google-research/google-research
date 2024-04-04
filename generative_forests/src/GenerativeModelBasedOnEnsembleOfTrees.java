// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class GenerativeModelBasedOnEnsembleOfTrees and related classes
 *****/

class GenerativeModelBasedOnEnsembleOfTrees implements Debuggable{
  public static String
      GENERATE_OBSERVATION_PICK_TREE_ITERATIVE = "GENERATE_OBSERVATION_PICK_TREE_ITERATIVE",
      GENERATE_OBSERVATION_PICK_TREE_RANDOM = "GENERATE_OBSERVATION_PICK_TREE_RANDOM";
    public static String GENERATE_OBSERVATION_PICK_TREE = GENERATE_OBSERVATION_PICK_TREE_RANDOM;
    // strategy to pick a tree for data generation (see paper)

    int name;

    boolean generative_forest, ensemble_of_generative_trees;
    
    Vector <Tree> trees;
    Dataset myDS;

    GenerativeModelBasedOnEnsembleOfTrees(Dataset ds, int n){
	name = n;
	myDS = ds;
	trees = null;

	generative_forest = true;
	ensemble_of_generative_trees = false;
    }

    public void generative_forest(){
	generative_forest = true;
	ensemble_of_generative_trees = false;
    }

    public void ensemble_of_generative_trees(){
	generative_forest = false;
	ensemble_of_generative_trees = true;
    }

    public void compute_probabilities_soft(){
	int i, j;
	int [] indexes;
	LocalEmpiricalMeasure lem;
	for (i=0;i<trees.size();i++){
	    indexes = new int[myDS.observations_from_file.size()];
	    for (j=0;j<myDS.observations_from_file.size();j++)
		indexes[j] = j;
	    lem = new LocalEmpiricalMeasure(myDS.observations_from_file.size());
	    lem.init_indexes(indexes);
	    lem.init_proportions(1.0);

	    trees.elementAt(i).root.p_reach = 1.0;
	    trees.elementAt(i).root.recursive_compute_probabilities_soft(lem);
	}
    }

    public static Vector <MeasuredSupportAtTupleOfNodes> ALL_SUPPORTS(Vector <MeasuredSupportAtTupleOfNodes> h){
	// h is initialized to singleton for a single observation
	// returns all subsets of supports resulting from the whole domain chopped off by all trees
	if ( (h.size() != 1) || (h.elementAt(0).done_for_all_supports()) )
	    Dataset.perror("GenerativeModelBasedOnEnsembleOfTrees.class :: bad initializer");
	
	Vector <MeasuredSupportAtTupleOfNodes> ret = new Vector<>(); // put here
	Vector <MeasuredSupportAtTupleOfNodes> current_mds_not_finished = new Vector<>();
	current_mds_not_finished.addElement(h.elementAt(0));
	
	Vector <MeasuredSupportAtTupleOfNodes> next_mds_not_finished;
	MeasuredSupportAtTupleOfNodes[] split;
	MeasuredSupportAtTupleOfNodes cur_mds;
	    
	int i, j;
	do{
	    next_mds_not_finished = new Vector<>();
	    for (i=0;i<current_mds_not_finished.size();i++){
		cur_mds = current_mds_not_finished.elementAt(i);
		split = cur_mds.split_for_all_supports();

		if (split == null){
		    if (cur_mds.done_for_all_supports())
			ret.addElement(cur_mds);
		    else
			next_mds_not_finished.addElement(cur_mds);
		}else{
		    for (j=0;j<2;j++)
			if (split[j].done_for_all_supports())
			    ret.addElement(split[j]);
			else
			    next_mds_not_finished.addElement(split[j]);
		}
	    }

	    if (next_mds_not_finished.size() > 0)
		current_mds_not_finished = next_mds_not_finished;
	}while(next_mds_not_finished.size() > 0);

	return ret;
    }

    public static void ALL_SUPPORTS_WITH_POSITIVE_MEASURE_GET(Vector <MeasuredSupportAtTupleOfNodes> h, Plotting pl, String geometric_primitive){
	// h is initialized to singleton for a single observation
	// returns all subsets of supports resulting from the whole domain chopped off by all trees
	if ( (h.size() != 1) || (h.elementAt(0).done_for_all_supports()) )
	    Dataset.perror("GenerativeModelBasedOnEnsembleOfTrees.class :: bad initializer");
	
	Vector <MeasuredSupportAtTupleOfNodes> current_mds_not_finished = new Vector<>();
	current_mds_not_finished.addElement(h.elementAt(0));
	
	Vector <MeasuredSupportAtTupleOfNodes> next_mds_not_finished;
	MeasuredSupportAtTupleOfNodes[] split;
	MeasuredSupportAtTupleOfNodes cur_mds;
	    
	int i, j;
	do{
	    next_mds_not_finished = new Vector<>();
	    for (i=0;i<current_mds_not_finished.size();i++){
		cur_mds = current_mds_not_finished.elementAt(i);
		split = cur_mds.split_for_all_supports_with_positive_measure();

		if (split == null){
		    if (cur_mds.done_for_all_supports()){
            if (geometric_primitive.equals(Plotting.LINES)) pl.add_line(cur_mds);
            else if (geometric_primitive.equals(Plotting.RECTANGLES)) pl.add_rectangle(cur_mds);
            else
              Dataset.perror(
                  "GenerativeModelBasedOnEnsembleOfTrees.class :: no such primitive: "
                      + geometric_primitive);
		    }else
			next_mds_not_finished.addElement(cur_mds);
		}else{
		    for (j=0;j<2;j++)
			if (split[j] != null){
			    if (split[j].done_for_all_supports()){
                if (geometric_primitive.equals(Plotting.LINES)) pl.add_line(split[j]);
                else if (geometric_primitive.equals(Plotting.RECTANGLES))
                  pl.add_rectangle(split[j]);
                else
                  Dataset.perror(
                      "GenerativeModelBasedOnEnsembleOfTrees.class :: no such primitive: "
                          + geometric_primitive);
			    }else
				next_mds_not_finished.addElement(split[j]);
			}
		}
	    }

	    if (next_mds_not_finished.size() > 0)
		current_mds_not_finished = next_mds_not_finished;
	}while(next_mds_not_finished.size() > 0);
    }


    public static Vector <MeasuredSupportAtTupleOfNodes> ALL_SUPPORTS_WITH_POSITIVE_MEASURE(Vector <MeasuredSupportAtTupleOfNodes> h){
	// h is initialized to singleton for a single observation
	// returns all subsets of supports resulting from the whole domain chopped off by all trees
	if ( (h.size() != 1) || (h.elementAt(0).done_for_all_supports()) )
	    Dataset.perror("GenerativeModelBasedOnEnsembleOfTrees.class :: bad initializer");
	
	Vector <MeasuredSupportAtTupleOfNodes> ret = new Vector<>(); // put here
	Vector <MeasuredSupportAtTupleOfNodes> current_mds_not_finished = new Vector<>();
	current_mds_not_finished.addElement(h.elementAt(0));
	
	Vector <MeasuredSupportAtTupleOfNodes> next_mds_not_finished;
	MeasuredSupportAtTupleOfNodes[] split;
	MeasuredSupportAtTupleOfNodes cur_mds;
	    
	int i, j;
	do{
	    next_mds_not_finished = new Vector<>();
	    for (i=0;i<current_mds_not_finished.size();i++){
		cur_mds = current_mds_not_finished.elementAt(i);
		split = cur_mds.split_for_all_supports_with_positive_measure();

		if (split == null){
		    if (cur_mds.done_for_all_supports())
			ret.addElement(cur_mds);
		    else
			next_mds_not_finished.addElement(cur_mds);
		}else{
		    for (j=0;j<2;j++)
			if (split[j] != null){
			    if (split[j].done_for_all_supports())
				ret.addElement(split[j]);
			    else
				next_mds_not_finished.addElement(split[j]);
			}
		}
	    }

	    if (next_mds_not_finished.size() > 0)
		current_mds_not_finished = next_mds_not_finished;
	}while(next_mds_not_finished.size() > 0);

	return ret;
    }

    public static Vector <WeightedSupportAtTupleOfNodes> ALL_SUPPORTS_COMPATIBLE_WITH_MISSING_DATA_ENSEMBLE_OF_GENERATIVE_TREES(Vector <WeightedSupportAtTupleOfNodes> h, Observation reference){
	// h is initialized to singleton for a single observation
	if ( (h.size() != 1) || (h.elementAt(0).done_for_missing_data_imputation()) )
	    Dataset.perror("GenerativeModelBasedOnEnsembleOfTrees.class :: bad initializer");
	
	Vector <WeightedSupportAtTupleOfNodes> ret = new Vector<>(); // put here
	Vector <WeightedSupportAtTupleOfNodes> current_mds_not_finished = new Vector<>();

	h.elementAt(0).prepare_for_missing_data_imputation();
	current_mds_not_finished.addElement(h.elementAt(0));
	
	Vector <WeightedSupportAtTupleOfNodes> next_mds_not_finished;
	WeightedSupportAtTupleOfNodes[] split;
	WeightedSupportAtTupleOfNodes cur_mds;

	int i, j;
	do{
	    next_mds_not_finished = new Vector<>();
	    for (i=0;i<current_mds_not_finished.size();i++){
		cur_mds = current_mds_not_finished.elementAt(i);
		split = cur_mds.split_for_missing_data_imputation_ensemble_of_generative_trees(reference);

		if (split == null){
		    if (cur_mds.done_for_missing_data_imputation())
			ret.addElement(cur_mds);
		    else
			next_mds_not_finished.addElement(cur_mds);
		}else{
		    for (j=0;j<2;j++)
			if (split[j] != null){
			    if (split[j].done_for_missing_data_imputation())
				ret.addElement(split[j]);
			    else
				next_mds_not_finished.addElement(split[j]);
			}
		}
	    }
	    
	    if (next_mds_not_finished.size() > 0)
		current_mds_not_finished = next_mds_not_finished;
	}while(next_mds_not_finished.size() > 0);
	
	return ret;
    }


    public static Vector <MeasuredSupportAtTupleOfNodes> ALL_SUPPORTS_COMPATIBLE_WITH_MISSING_DATA_GENERATIVE_FOREST(Vector <MeasuredSupportAtTupleOfNodes> h, Observation reference){
	// h is initialized to singleton for a single observation
	if ( (h.size() != 1) || (h.elementAt(0).done_for_missing_data_imputation()) )
	    Dataset.perror("GenerativeModelBasedOnEnsembleOfTrees.class :: bad initializer");
	
	Vector <MeasuredSupportAtTupleOfNodes> ret = new Vector<>(); // put here
	Vector <MeasuredSupportAtTupleOfNodes> current_mds_not_finished = new Vector<>();
	current_mds_not_finished.addElement(h.elementAt(0));

	h.elementAt(0).prepare_for_missing_data_imputation();
	
	Vector <MeasuredSupportAtTupleOfNodes> next_mds_not_finished;
	MeasuredSupportAtTupleOfNodes[] split;
	MeasuredSupportAtTupleOfNodes cur_mds;
	    
	int i, j;
	do{
	    next_mds_not_finished = new Vector<>();
	    for (i=0;i<current_mds_not_finished.size();i++){
		cur_mds = current_mds_not_finished.elementAt(i);
		split = cur_mds.split_for_missing_data_imputation_generative_forest(reference);

		if (split == null){
          if (!cur_mds.generative_support.local_empirical_measure.contains_indexes())
            Dataset.perror(
                "MeasuredSupportAtTupleOfNodes.class :: null split & no indexes in " + cur_mds);

		    if (cur_mds.done_for_missing_data_imputation())
			ret.addElement(cur_mds);
		    else
			next_mds_not_finished.addElement(cur_mds);
		}else{
		    for (j=0;j<2;j++)
			if (split[j] != null){
              if (!cur_mds.generative_support.local_empirical_measure.contains_indexes())
                Dataset.perror(
                    "MeasuredSupportAtTupleOfNodes.class :: split["
                        + j
                        + "]: no indexes in "
                        + split[j]);

			    if (split[j].done_for_missing_data_imputation())
				ret.addElement(split[j]);
			    else
				next_mds_not_finished.addElement(split[j]);
			}
		}
	    }

	    if (next_mds_not_finished.size() > 0)
		current_mds_not_finished = next_mds_not_finished;
	}while(next_mds_not_finished.size() > 0);

	return ret;
    }

    
    public void impute(Observation oo_at_the_max, Observation oo_sampling_density, int index_obs){
	if (generative_forest)
	    impute_as_generative_forest(oo_at_the_max, oo_sampling_density, index_obs);
	else if (ensemble_of_generative_trees)
	    impute_as_ensemble_of_generative_trees(oo_at_the_max, oo_sampling_density, index_obs);
	else
	    Dataset.perror("GenerativeModelBasedOnEnsembleOfTrees.class :: no imputation method");
    }
    
    public void impute_as_generative_forest(Observation oo_at_the_max, Observation oo_sampling_density, int index_obs){
	// index_obs used to remove the observation from the computation of the support if it were part of training

	Observation reference = null;

    if (oo_at_the_max != null) reference = Observation.copyOf(oo_at_the_max);
    else if (oo_sampling_density != null) reference = Observation.copyOf(oo_sampling_density);
    else
      Dataset.perror(
          "GenerativeModelBasedOnEnsembleOfTrees.class :: both observations to impute are null");

    if (!reference.contains_unknown_values())
      Dataset.perror(
          "GenerativeModelBasedOnEnsembleOfTrees.class :: imputing an observation "
              + reference
              + " without missing values");

	MeasuredSupportAtTupleOfNodes init_mds = new MeasuredSupportAtTupleOfNodes(this, true, false, index_obs);
	init_mds.generative_support.local_empirical_measure.init_proportions(1.0);

	init_mds.squash_for_missing_data_imputation(reference);
	
	Vector <MeasuredSupportAtTupleOfNodes> m = new Vector<>();
	m.addElement(init_mds);
	
	Vector <MeasuredSupportAtTupleOfNodes> mds = GenerativeModelBasedOnEnsembleOfTrees.ALL_SUPPORTS_COMPATIBLE_WITH_MISSING_DATA_GENERATIVE_FOREST(m, reference);

	MeasuredSupportAtTupleOfNodes target_mds_at_the_max = null, target_mds_sampling_density = null;
	Feature target_feature;

	// compute densities
	double [] densities = new double [mds.size()];
	double tot = 0.0, dmax = -1.0, drand = -1.0, dval, cw;
	int i, j, index_at_the_max = -1, index_sampling_density = -1;

	for (i=0;i<mds.size();i++){
	    cw = (double) mds.elementAt(i).generative_support.local_empirical_measure.total_weight;
	    tot += cw; 
	}

	if (oo_at_the_max != null){
	    for (i=0;i<mds.size();i++){
		cw = (double) mds.elementAt(i).generative_support.local_empirical_measure.total_weight;	    
		densities[i] = (cw / tot) / (double) mds.elementAt(i).generative_support.support.volume;
		if ( (i==0) || (densities[i] > dmax) )
		    dmax = densities[i];
	    }

	    Vector <Integer> indexes_max = new Vector<>();
	    for (i=0;i<mds.size();i++)
		if (densities[i] == dmax)
		    indexes_max.addElement(new Integer(i));

	    index_at_the_max = indexes_max.elementAt(Statistics.R.nextInt(indexes_max.size())).intValue();
	    target_mds_at_the_max = mds.elementAt(index_at_the_max);
	}

	if (oo_sampling_density != null){
	    drand = tot * Statistics.R.nextDouble();
	    cw = 0.0;
	    i = 0;
	    cw += (double) mds.elementAt(i).generative_support.local_empirical_measure.total_weight;
	    while ( (i < mds.size() - 1) && (drand > cw) ){
		i++;
		cw += (double) mds.elementAt(i).generative_support.local_empirical_measure.total_weight;
	    }
      if (drand > cw)
        Dataset.perror(
            "GenerativeModelBasedOnEnsembleOfTrees.class :: bin chosen too far ("
                + drand
                + " > "
                + cw
                + ")");

	    index_sampling_density = i;
	    target_mds_sampling_density = mds.elementAt(index_sampling_density);
	}

	// impute using target_mds
	int index;

	if (oo_at_the_max != null){
	    for (i=0;i<reference.typed_features.size();i++)
		if (FeatureValue.IS_UNKNOWN(reference.typed_features.elementAt(i))){
		    target_feature = target_mds_at_the_max.generative_support.support.feature(i);
		    if (Feature.IS_NOMINAL(target_feature.type)){
			index = Statistics.R.nextInt(target_feature.modalities.size());
			oo_at_the_max.typed_features.setElementAt(new FeatureValue(target_feature.modalities.elementAt(index)), i);
		    }else if (Feature.IS_INTEGER(target_feature.type)){
			index = Statistics.R.nextInt(target_feature.imax - target_feature.imin + 1);
			oo_at_the_max.typed_features.setElementAt(new FeatureValue(target_feature.imin + index), i);
		    }else if (Feature.IS_CONTINUOUS(target_feature.type)){
			dval = (Statistics.R.nextDouble())*(target_feature.dmax - target_feature.dmin);
			oo_at_the_max.typed_features.setElementAt(new FeatureValue(target_feature.dmin + dval), i);
          } else
            Dataset.perror(
                "Generator_Tree.class :: Feature " + target_feature + "'s type not known");
		}
	}

	if (oo_sampling_density != null){
	    for (i=0;i<reference.typed_features.size();i++)
		if (FeatureValue.IS_UNKNOWN(reference.typed_features.elementAt(i))){
		    target_feature = target_mds_sampling_density.generative_support.support.feature(i);
		    if (Feature.IS_NOMINAL(target_feature.type)){
			index = Statistics.R.nextInt(target_feature.modalities.size());
			oo_sampling_density.typed_features.setElementAt(new FeatureValue(target_feature.modalities.elementAt(index)), i);
		    }else if (Feature.IS_INTEGER(target_feature.type)){
			index = Statistics.R.nextInt(target_feature.imax - target_feature.imin + 1);
			oo_sampling_density.typed_features.setElementAt(new FeatureValue(target_feature.imin + index), i);
		    }else if (Feature.IS_CONTINUOUS(target_feature.type)){
			dval = (Statistics.R.nextDouble())*(target_feature.dmax - target_feature.dmin);
			oo_sampling_density.typed_features.setElementAt(new FeatureValue(target_feature.dmin + dval), i);
          } else
            Dataset.perror(
                "Generator_Tree.class :: Feature " + target_feature + "'s type not known");
		}
	}
    }

    public void impute_as_ensemble_of_generative_trees(Observation oo_at_the_max, Observation oo_sampling_density, int index_obs){
	// index_obs used to remove the observation from the computation of the support if it were part of training
	
	Observation reference = null;

    if (oo_at_the_max != null) reference = Observation.copyOf(oo_at_the_max);
    else if (oo_sampling_density != null) reference = Observation.copyOf(oo_sampling_density);
    else
      Dataset.perror(
          "GenerativeModelBasedOnEnsembleOfTrees.class :: both observations to impute are null");

    if (!reference.contains_unknown_values())
      Dataset.perror(
          "GenerativeModelBasedOnEnsembleOfTrees.class :: imputing an observation "
              + reference
              + " without missing values");

	WeightedSupportAtTupleOfNodes init_mds = new WeightedSupportAtTupleOfNodes(this);
	init_mds.generative_support.total_weight = 1.0;
	init_mds.squash_for_missing_data_imputation(reference);
	
	Vector <WeightedSupportAtTupleOfNodes> m = new Vector<>();
	m.addElement(init_mds);
	
	Vector <WeightedSupportAtTupleOfNodes> mds =  GenerativeModelBasedOnEnsembleOfTrees.ALL_SUPPORTS_COMPATIBLE_WITH_MISSING_DATA_ENSEMBLE_OF_GENERATIVE_TREES(m, reference);

	WeightedSupportAtTupleOfNodes target_mds_at_the_max = null, target_mds_sampling_density = null;
	Feature target_feature;

	// compute densities
	double [] densities = new double [mds.size()]; // NOT NORMALIZED (FASTER)
	double dmax = -1.0, drand = -1.0, dval, cw, tot = 0.0;
	int i, index_at_the_max = -1, index_sampling_density = -1;

	if (oo_sampling_density != null)
	    for (i=0;i<mds.size();i++){
		cw = (double) mds.elementAt(i).generative_support.total_weight;
		tot += cw; 
	    }

	if (oo_at_the_max != null){
	    for (i=0;i<mds.size();i++){	    
		densities[i] = (double) mds.elementAt(i).generative_support.total_weight / (double) mds.elementAt(i).generative_support.support.volume;
		if ( (i==0) || (densities[i] > dmax) )
		    dmax = densities[i];
	    }

	    Vector <Integer> indexes_max = new Vector<>();
	    for (i=0;i<mds.size();i++)
		if (densities[i] == dmax)
		    indexes_max.addElement(new Integer(i));

	    index_at_the_max = indexes_max.elementAt(Statistics.R.nextInt(indexes_max.size())).intValue();
	    target_mds_at_the_max = mds.elementAt(index_at_the_max);
	}

	if (oo_sampling_density != null){
	    drand = tot * Statistics.R.nextDouble();
	    cw = 0.0;
	    i = 0;
	    cw += (double) mds.elementAt(i).generative_support.total_weight; 
	    while ( (i < mds.size() - 1) && (drand > cw) ){
		i++;
		cw += (double) mds.elementAt(i).generative_support.total_weight;
	    }
      if (drand > cw)
        Dataset.perror(
            "GenerativeModelBasedOnEnsembleOfTrees.class :: bin chosen too far ("
                + drand
                + " > "
                + cw
                + ")");

	    index_sampling_density = i;
	    target_mds_sampling_density = mds.elementAt(index_sampling_density);
	}

	// impute using target_mds
	int index;

	if (oo_at_the_max != null){
	    for (i=0;i<reference.typed_features.size();i++)
		if (FeatureValue.IS_UNKNOWN(reference.typed_features.elementAt(i))){
		    target_feature = target_mds_at_the_max.generative_support.support.feature(i);
		    if (Feature.IS_NOMINAL(target_feature.type)){
			index = Statistics.R.nextInt(target_feature.modalities.size());
			oo_at_the_max.typed_features.setElementAt(new FeatureValue(target_feature.modalities.elementAt(index)), i);
		    }else if (Feature.IS_INTEGER(target_feature.type)){
			index = Statistics.R.nextInt(target_feature.imax - target_feature.imin + 1);
			oo_at_the_max.typed_features.setElementAt(new FeatureValue(target_feature.imin + index), i);
		    }else if (Feature.IS_CONTINUOUS(target_feature.type)){
			dval = (Statistics.R.nextDouble())*(target_feature.dmax - target_feature.dmin);
			oo_at_the_max.typed_features.setElementAt(new FeatureValue(target_feature.dmin + dval), i);
          } else
            Dataset.perror(
                "Generator_Tree.class :: Feature " + target_feature + "'s type not known");
		}
	}

	if (oo_sampling_density != null){
	    for (i=0;i<reference.typed_features.size();i++)
		if (FeatureValue.IS_UNKNOWN(reference.typed_features.elementAt(i))){
		    target_feature = target_mds_sampling_density.generative_support.support.feature(i);
		    if (Feature.IS_NOMINAL(target_feature.type)){
			index = Statistics.R.nextInt(target_feature.modalities.size());
			oo_sampling_density.typed_features.setElementAt(new FeatureValue(target_feature.modalities.elementAt(index)), i);
		    }else if (Feature.IS_INTEGER(target_feature.type)){
			index = Statistics.R.nextInt(target_feature.imax - target_feature.imin + 1);
			oo_sampling_density.typed_features.setElementAt(new FeatureValue(target_feature.imin + index), i);
		    }else if (Feature.IS_CONTINUOUS(target_feature.type)){
			dval = (Statistics.R.nextDouble())*(target_feature.dmax - target_feature.dmin);
			oo_sampling_density.typed_features.setElementAt(new FeatureValue(target_feature.dmin + dval), i);
          } else
            Dataset.perror(
                "Generator_Tree.class :: Feature " + target_feature + "'s type not known");
		}
	}
    }
    
    public void init(int nb_trees){
	trees = new Vector <>();
	int i;
	for (i=0;i<nb_trees;i++){
	    trees.add(new Tree(this, i));
	}
    }

    public String toString(){
	if (trees == null)
	    return "(null)";

	int i;
	String v = "";
	for (i=0;i<trees.size();i++){
      v +=
          "---------------------------------------------------------------------------------------------------------------\n";
	    v += trees.elementAt(i);
	}
    v +=
        "---------------------------------------------------------------------------------------------------------------\n";
	return v;
    }

    public int generate_one_observation_pick_tree(Vector<Integer> available_trees_indexes){
    if (GenerativeModelBasedOnEnsembleOfTrees.GENERATE_OBSERVATION_PICK_TREE.equals(
        GenerativeModelBasedOnEnsembleOfTrees.GENERATE_OBSERVATION_PICK_TREE_ITERATIVE)) return 0;
    else if (GenerativeModelBasedOnEnsembleOfTrees.GENERATE_OBSERVATION_PICK_TREE.equals(
        GenerativeModelBasedOnEnsembleOfTrees.GENERATE_OBSERVATION_PICK_TREE_RANDOM))
      return Statistics.R.nextInt(available_trees_indexes.size());
    else
      Dataset.perror(
          "GenerativeModelBasedOnEnsembleOfTrees.class :: no such tree choice method as "
              + GenerativeModelBasedOnEnsembleOfTrees.GENERATE_OBSERVATION_PICK_TREE);

	return -1;
    }

    public Observation generate_one_observation(){
	MeasuredSupport gs = new MeasuredSupport(this, -1);
	gs.local_empirical_measure.init_proportions(1.0);
	
	Tree tree_pick;
	int i, iter = 0, index, dumi = 0;

	double local_density;
	
	for (i=0;i<trees.size();i++)
	    trees.elementAt(i).init_generation();

	// pick a tree (Cf paper for strategies)

	Vector<Integer> available_trees_indexes = new Vector<>();
	for (i=0;i<trees.size();i++)
	    if (!trees.elementAt(i).generation_done())
		available_trees_indexes.addElement(new Integer(i));

    if (available_trees_indexes.size() == 0)
      Dataset.perror(
          "GenerativeModelBasedOnEnsembleOfTrees.class :: no non-trivial generation possible, only"
              + " roots = leaves");

	do{
	    index = generate_one_observation_pick_tree(available_trees_indexes);
	    tree_pick = trees.elementAt(index);

	    gs = tree_pick.update_star_node_and_support(gs);
	    if (trees.elementAt(index).generation_done())
		available_trees_indexes.removeElementAt(index);
	}while(available_trees_indexes.size() > 0);

    if (gs.support.volume == 0.0)
      Dataset.perror(
          "GenerativeModelBasedOnEnsembleOfTrees.class :: generating an observation in a support"
              + " with zero volume ("
              + gs.support
              + ")");

	local_density = ((double) gs.local_empirical_measure.observations_indexes.length / (double) myDS.observations_from_file.size()) / gs.support.volume;

    if (Double.isInfinite(local_density))
      Dataset.perror(
          "GenerativeModelBasedOnEnsembleOfTrees.class :: Infinite local_density on local support ("
              + gs.support
              + ")");

	Vector <String> coordinates = new Vector <>();
	Feature f;

	int ival;
	double dval;
	
	for (i=0;i<gs.support.dim();i++){
	    f = gs.support.feature(i);
	    if (f.type.equals(Feature.CONTINUOUS)){
		dval = f.dmin + ((Statistics.R.nextDouble())*(f.dmax - f.dmin));

		coordinates.addElement(new String("" + dval));
	    }else if (f.type.equals(Feature.INTEGER)){
		ival = f.imin + (Statistics.R.nextInt(f.imax - f.imin + 1));
		
		coordinates.addElement(new String("" + ival));
	    }else if (f.type.equals(Feature.NOMINAL))
		coordinates.addElement(new String(f.modalities.elementAt(Statistics.R.nextInt(f.modalities.size()))));
	}

	Observation oo = new Observation(-1, coordinates, myDS.features, -1.0, local_density);

	return oo;
    }

    public Vector<Observation> generate_sample_with_density(int nex){
	int i;
	Vector<Observation> set = new Vector<>();
	Observation oo;
	for (i=0;i<nex;i++){
	    if (i%(nex/10) == 0)
		    System.out.print((i/(nex/10))*10 + "% ");
	    
	    oo = generate_one_observation();
	    set.addElement(oo);
	}
	return set;
    }

    public static Vector <MeasuredSupportAtTupleOfNodes> ALL_SUPPORTS_COMPATIBLE_WITH_OBSERVATION_GENERATIVE_FOREST(Vector <MeasuredSupportAtTupleOfNodes> h, Observation reference){
	// h is initialized to singleton for a single observation
	if (h.size() != 1)
	    Dataset.perror("GenerativeModelBasedOnEnsembleOfTrees.class :: bad initializer");
	
	Vector <MeasuredSupportAtTupleOfNodes> ret = new Vector<>(); // put here
	Vector <MeasuredSupportAtTupleOfNodes> current_mds_not_finished = new Vector<>();
	current_mds_not_finished.addElement(h.elementAt(0));

	h.elementAt(0).prepare_for_missing_data_imputation();
	
	Vector <MeasuredSupportAtTupleOfNodes> next_mds_not_finished;
	MeasuredSupportAtTupleOfNodes[] split;
	MeasuredSupportAtTupleOfNodes cur_mds;
	    
	int i, j;
	do{
	    next_mds_not_finished = new Vector<>();
	    for (i=0;i<current_mds_not_finished.size();i++){
		cur_mds = current_mds_not_finished.elementAt(i);
		split = cur_mds.split_to_find_all_supports_of_reference_in_generative_forest(reference);

		if (split == null){
          if (!cur_mds.generative_support.local_empirical_measure.contains_indexes())
            Dataset.perror(
                "MeasuredSupportAtTupleOfNodes.class :: null split & no indexes in " + cur_mds);

		    if (cur_mds.done_for_all_supports())
			ret.addElement(cur_mds);
		    else
			next_mds_not_finished.addElement(cur_mds);
		}else{
		    for (j=0;j<2;j++)
			if (split[j] != null){
              if (!cur_mds.generative_support.local_empirical_measure.contains_indexes())
                Dataset.perror(
                    "MeasuredSupportAtTupleOfNodes.class :: split["
                        + j
                        + "]: no indexes in "
                        + split[j]);

			    if (split[j].done_for_missing_data_imputation())
				ret.addElement(split[j]);
			    else
				next_mds_not_finished.addElement(split[j]);
			}
		}
	    }

	    if (next_mds_not_finished.size() > 0)
		current_mds_not_finished = next_mds_not_finished;
	}while(next_mds_not_finished.size() > 0);

	return ret;
    }    

    public double [][] expected_density_estimation_output(){
	
	FileWriter f;
	int i,j = 0,k, n_test = 0;

	// creating vector of observations from test file
	FileReader e;
	BufferedReader br;
	StringTokenizer t;

	boolean feature_names_line = false;
	Vector <String> current_observation;
	String dum, n, ty;

	Vector <Observation> observations_test = new Vector<>();
	Observation ee;
	int errfound = 0;

	try{
	    e = new FileReader(myDS.myDomain.myW.path_and_name_of_test_dataset);
	    br = new BufferedReader(e);

	    while ( (dum=br.readLine()) != null){
		if ( (dum.length()>1) && (!dum.substring(0,Dataset.KEY_COMMENT.length()).equals(Dataset.KEY_COMMENT)) ){
		    if (!feature_names_line)
			feature_names_line = true; // first line must also be features
		    else{
			// records all following values; checks sizes comply
			current_observation = new Vector<>();
			t = new StringTokenizer(dum,Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
            if (t.countTokens() != myDS.number_features_total)
              Dataset.perror(
                  "Wrapper.class :: Observation string + "
                      + dum
                      + " does not have "
                      + myDS.number_features_total
                      + " features in *test file*");
			while(t.hasMoreTokens())
			    current_observation.addElement(t.nextToken());

			ee = new Observation(j, current_observation, myDS.features, 1.0);
			errfound += ee.checkAndCompleteFeatures(myDS.features);
			
			observations_test.addElement(ee);

			j++;
		    }
		}
	    }
	}catch(IOException eee){
      System.out.println(
          "Problem loading "
              + myDS.myDomain.myW.path_and_name_of_test_dataset
              + " file --- Check the access to file");
	    System.exit(0);
	}

    if (errfound > 0)
      Dataset.perror(
          "Wrapper.class :: found "
              + errfound
              + " errs for feature domains in observations for *test file*. Please correct features"
              + " in file. ");

	n_test = j;
	for (i=0;i<n_test;i++)
	    observations_test.elementAt(i).weight = 1.0 / (double) n_test;

	MeasuredSupportAtTupleOfNodes init_mds;
	Vector <MeasuredSupportAtTupleOfNodes> m;
	Vector <MeasuredSupportAtTupleOfNodes> mds;

	double tot_w, tot_v, log_likelihood, expected_log_likelihood = 0.0, likelihood, expected_likelihood = 0.0;

	double [] all_log_likelihoods = new double [n_test];
	double [] all_likelihoods = new double [n_test];
	double [] averagestd_likelihoods = new double[2];
	double [] averagestd_log_likelihoods = new double[2];
	double [][] averagestd = new double[2][];
	averagestd[0] = new double[2];
	averagestd[1] = new double[2];
	
	for (i=0;i<n_test;i++){
	    ee = observations_test.elementAt(i);
	    	    
	    init_mds = new MeasuredSupportAtTupleOfNodes(this, true, false, -1);
	    init_mds.generative_support.local_empirical_measure.init_proportions(1.0);

	     m = new Vector<>();
	     m.addElement(init_mds);
	     mds = GenerativeModelBasedOnEnsembleOfTrees.ALL_SUPPORTS_COMPATIBLE_WITH_OBSERVATION_GENERATIVE_FOREST(m, ee);

	     tot_w = 0.0;
	     tot_v = 0.0;

	     for (j=0;j<mds.size();j++){
		 tot_w += (double) mds.elementAt(j).generative_support.local_empirical_measure.total_weight;
		 tot_v += (double) mds.elementAt(j).generative_support.support.volume;
	     }


	     all_likelihoods[i] = (tot_w / (double) myDS.observations_from_file.size()) / tot_v;
	     all_log_likelihoods[i] =  Math.log(all_likelihoods[i]);
	}

	Statistics.avestd(all_log_likelihoods, averagestd_log_likelihoods);
	Statistics.avestd(all_likelihoods, averagestd_likelihoods);

	averagestd[0] = averagestd_likelihoods;
	averagestd[1] = averagestd_log_likelihoods;
	
	return averagestd;
    }


}

