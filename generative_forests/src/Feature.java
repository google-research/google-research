// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Support
 * support related stuff
 *****/

class Support implements Debuggable{
    private Vector <Feature> features;

    double volume = -1.0;
    double weight_uniform_distribution = -1.0;

    Support(){
	features = null;
    }
    
    Support(Vector <Feature> vf, Dataset ds){
	features = new Vector <>();
	int i;
	double wud = 1.0, v = 1.0;
	
	for (i=0;i<vf.size();i++){
	    if (vf.elementAt(i) != null){
		features.addElement(Feature.copyOf(vf.elementAt(i)));
		v *= features.elementAt(i).length();
		wud *= features.elementAt(i).length() / ds.features.elementAt(i).length();
	    }else
		features.addElement(null); // option for Missing Data Imputation
	}

	volume = v;
	weight_uniform_distribution = wud;
    }
    
    public String toString(){
	int i;
	String ret = "";
	for (i = 0; i<features.size();i++){
	    if (features.elementAt(i) == null)
		ret += "[ null ]";  // option for Missing Data Imputation
	    else
		ret += features.elementAt(i).toShortString();
	    if (i<features.size() - 1)
		ret += " X ";
	}
	ret += "{{" + volume + "}}{" + weight_uniform_distribution + "}";
	return ret;
    }

    public static Support[] SPLIT_SUPPORT(Dataset ds, Support s, FeatureTest ft, int feature_split_index){
	if ( (s.volume == -1.0) || (s.weight_uniform_distribution == -1.0) )
	    Dataset.perror("Support.class :: SPLIT_SUPPORT cannot split, negative weights");
	
	Feature feature_in_measured_support = s.feature(feature_split_index);
	
	Support[] ret = new Support[2];
	FeatureTest f = FeatureTest.copyOf(ft, feature_in_measured_support);
	Feature [] split_feat;

	String tvb = f.check_trivial_branching(ds, feature_in_measured_support, true);

	if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_LEFT)){
	    ret[0] = Support.copyOf(s);
		
	    ret[1] = null; //right
	}else if (tvb.equals(FeatureTest.TRIVIAL_BRANCHING_RIGHT)){
	    ret[0] = null; //left
	    
	    ret[1] = Support.copyOf(s);
	}else{
	    ret[0] = Support.copyOf(s);
	    ret[1] = Support.copyOf(s);

	    split_feat = f.split_feature(ds, feature_in_measured_support, false, false);

	    ret[0].setFeatureAt(split_feat[0], feature_split_index);	    
	    ret[1].setFeatureAt(split_feat[1], feature_split_index);
	}
	    
	return ret;
    }

    public Support[] split_support(int feature_index, Feature [] new_features){
	// split support at a leaf, ensuring consistency

	if ( (volume == -1.0) || (weight_uniform_distribution == -1.0) )
	    Dataset.perror("Support.class :: split_support cannot split, negative weights");
	Feature.TEST_UNION(features.elementAt(feature_index), new_features[0], new_features[1]);

	Support [] ret = new Support[2];
	ret[0] = Support.copyOf(this);
	ret[1] = Support.copyOf(this);

	ret[0].setFeatureAt(new_features[0], feature_index);
	ret[1].setFeatureAt(new_features[1], feature_index);
	
	return ret;
    }

    public static Support copyOf(Support vc){
	if ( (vc.volume == -1.0) || (vc.weight_uniform_distribution == -1.0) )
	    Dataset.perror("Support.class :: copyOf cannot copy, negative weights");

	Support ret = new Support();
	
	ret.features = new Vector <>();
	int i;
	for (i=0;i<vc.features.size();i++)
	    if (vc.features.elementAt(i) != null){
		ret.features.addElement(Feature.copyOf(vc.features.elementAt(i)));
	    }else
		ret.features.addElement(null);
	
	ret.volume = vc.volume;
	ret.weight_uniform_distribution = vc.weight_uniform_distribution;
	
	return ret;
    }

    public static Support CAP(Vector <Node> nodes, boolean ensure_non_zero_measure, Dataset ds){
	// cheaper option to compute the intersection of supports
	// proceeds by feature; if one \cap returns null, returns null

	Feature [] f_cur;
	Vector <Feature> ret = new Vector <>();
	Feature f;
	int i, j;
	for (i=0;i<nodes.elementAt(0).node_support.dim();i++){
	    f_cur = new Feature[nodes.size()];
	    for (j=0;j<nodes.size();j++){
		if (nodes.elementAt(j).node_support.dim() != ds.features.size())
		    Dataset.perror("Support.class :: cannot compute CAP");
		f_cur[j] = nodes.elementAt(j).node_support.feature(i);
	    }
	    f = Feature.CAP(f_cur, ensure_non_zero_measure);
	    if (f == null)
		return null;
	    else
		ret.addElement(f);
	}
	return new Support(ret, ds);
    }

    public Support cap(Support t, boolean ensure_non_zero_measure){
	// return the intersection feature: EXPENSIVE REPLACE BY CAP

	Support ret = new Support();
	ret.volume = volume;
	ret.weight_uniform_distribution = weight_uniform_distribution;
	
	ret.features = new Vector <>();
	int i;
	Feature df;
	double rat;
	for (i=0;i<dim();i++){
	    df = features.elementAt(i).cap(t.feature(i), ensure_non_zero_measure);
	    if (df == null)
		return null;
	    ret.features.addElement(df);

	    if (features.elementAt(i).length() == 0.0)
		Dataset.perror("Support.class :: cannot cap support, /0");

	    rat = df.length() / features.elementAt(i).length();
	    
	    ret.volume *= rat;
	    ret.weight_uniform_distribution *= rat;	    
	}

	return ret;
    }

    
    public static boolean SUPPORT_INTERSECTION_IS_EMPTY(Vector <Node> nodes, boolean ensure_non_zero_measure, Dataset ds){
	Support ret = Support.CAP(nodes, ensure_non_zero_measure, ds);

	if (ret == null)
	    return true;
	return false;
    }


    public Feature feature(int i){
	return features.elementAt(i);
    }

    public void setFeatureAt(Feature f, int i){
	double rat;

	if (features.elementAt(i) == null)
	    Dataset.perror("Support.class :: trouble in missing data imputation");

	if (features.elementAt(i).length() == 0.0)
	    Dataset.perror("Feature.class :: cannot uppdate vol / wud: /0");
	
	rat = f.length() / features.elementAt(i).length();
	volume *= rat;
	weight_uniform_distribution *= rat;	
	features.setElementAt(f, i);
    }

    public void setNullFeatureAt(int i, Dataset ds){
	double rat;

	if (features.elementAt(i).length() == 0.0)
	    Dataset.perror("Feature.class :: cannot uppdate vol / wud: /0");
	
	rat = ds.features.elementAt(i).length() / features.elementAt(i).length();
	volume /= features.elementAt(i).length();
	weight_uniform_distribution *= rat;
	
	features.setElementAt(null, i);
    }
    
    public int dim(){
	return features.size();
    }

    public boolean is_subset_of(Support t){
	//returns true iff this \subseteq t
	if (dim() != t.dim())
	    Dataset.perror("Feature.class :: supports not comparable");
	
	int i;
	for (i=0;i<features.size();i++)
	    if (!Feature.IS_SUBFEATURE(features.elementAt(i), t.feature(i)))
		return false;
	return true;
    }


    public boolean observation_known_values_in_support(Observation e){
	// return true iff all SPECIFIED attributes in e are in the corresponding support feature's domain
	// ignores unspecified attributes

	int i;
	for (i=0;i<dim();i++){
	    if ( (!e.typed_features.elementAt(i).type.equals(Feature.UNKNOWN_VALUE)) && (!Feature.OBSERVATION_MATCHES_FEATURE(e, features.elementAt(i), i)) )
		return false;
	}
	return true;
    }
}

/**************************************************************************************************************************************
 * Class FeatureTest
 * provides encapsulation for sets relevant to feature tests
 * a FeatureTest is used to compute the branching probabilities IF an observation's this.name is unknown
 *****/

class FeatureTest implements Debuggable{
  // private to prevent direct access
  public static String TRIVIAL_BRANCHING_LEFT = "TRIVIAL_BRANCHING_LEFT",
      TRIVIAL_BRANCHING_RIGHT = "TRIVIAL_BRANCHING_RIGHT",
      NO_TRIVIAL_BRANCHING = "NO_TRIVIAL_BRANCHING";

    //private Feature myFeature;
    // removed and replaced by the name of the feature
    public String name;
    
    private double double_test;
    private int int_test;
    private BoolArray boolarray_to_string_test;
    // do not store Strings but booleans:
    // MUST HAVE LENGTH = the domain of the feature in Dataset

    String type;

    public static boolean HAS_AT_LEAST_ONE_FEATURE_TEST(Feature f, Dataset ds){
	if (!IS_SPLITTABLE(f))
	    return false;

	if ( (Feature.IS_CONTINUOUS(f.type)) && (f.dmax - f.dmin <= 0.0) )
	    return false;

	if ( (Feature.IS_INTEGER(f.type)) && (f.imax - f.imin <= 0) )
	    return false;

	if ( (Feature.IS_NOMINAL(f.type)) && (f.modalities.size() <= 1) )
	    return false;

	return true;
    }

    public static Vector<FeatureTest> ALL_FEATURE_TESTS(Feature f, Dataset ds){
	// if continuous, list of evenly spaced ties
	// if nominal, list of partial non-empty subsets of the whole set
	// if integer, list of integers

	if (!IS_SPLITTABLE(f))
	    return null;

	Vector <FeatureTest> v = new Vector<FeatureTest> ();
	int i, j;
	if (Feature.IS_CONTINUOUS(f.type)){
	    if (f.dmax - f.dmin <= 0.0){
		v = null;
	    }else{		
		double vmin = f.dmin;
		double vmax = f.dmax;
		double delta = (vmax - vmin) / ( (double) (Feature.NUMBER_CONTINUOUS_TIES + 1) );
		double vcur = vmin + delta;
		for (i=0;i<Feature.NUMBER_CONTINUOUS_TIES;i++){
		    v.add(new FeatureTest(vcur, f));
		    vcur += delta;
		}
	    }
	}else if (Feature.IS_INTEGER(f.type)){
	    if (f.imax - f.imin <= 0){
		v = null;
	    }else{
		int vmin = f.imin;
		int nvals = f.imax - f.imin;
		for (i=0;i<nvals;i++){
		    v.add(new FeatureTest(vmin + i, f));
		}
	    }
	}else if (Feature.IS_NOMINAL(f.type)){
	    Vector <BoolArray> all_boolarrays_not_rescaled;
	    BoolArray dumba, baind;
	    
	    Feature reference_in_ds = ds.features.elementAt(ds.indexOfFeature(f.name));
	    
	    all_boolarrays_not_rescaled = Utils.ALL_NON_TRIVIAL_BOOLARRAYS(f.modalities.size(), Utils.GUESS_MAX_TRUE_IN_BOOLARRAY(f.modalities.size()));

	    //making sure the BoolArrays tests are re-indexed on DS feature modalities
	    if (all_boolarrays_not_rescaled.size() > 0){
		for (i=0;i<all_boolarrays_not_rescaled.size();i++){
		    dumba = new BoolArray(reference_in_ds.modalities.size());
		    baind = all_boolarrays_not_rescaled.elementAt(i);
		    for (j=0;j<baind.size();j++)
			if (baind.get(j))
			    dumba.set(reference_in_ds.modalities.indexOf(f.modalities.elementAt(j)), true);
		    v.add(new FeatureTest(dumba, f));
		}
	    }else
		v = null;
	}

	return v;
    }


    public static String DISPLAY_TESTS(Vector<FeatureTest> ft, boolean show_all){
	int max_display = 5;
	if (show_all)
	    max_display = ft.size();
	
	int i, j;
	String v = "{";
	Vector <String> dv;
	if (ft != null){
	    if (ft.size() == 0)
		Dataset.perror("Feature.class :: avoid empty but non null test sets");
	    for (i =0;i<ft.size();i++){
		if (i < max_display){
		    v += ft.elementAt(i);
		    if (i<ft.size()-1)
			v += ", ";
		}else{
		    v += "... ";
		    break;
		}
	    }
	}
	v += "}";
	return v;
    }
    
    public static boolean IS_SPLITTABLE(Feature f){
	if (Feature.IS_NOMINAL(f.type)){
	    if (f.modalities == null)
		return false;
	    if (f.modalities.size() <= 1)
		return false;
	}else if (Feature.IS_CONTINUOUS(f.type)){
	    if (f.dmin >= f.dmax)
		return false;
	}else if (Feature.IS_INTEGER(f.type)){
	    if (f.imin >= f.imax - 1)
		return false;
	}

	return true;
    }
    
    FeatureTest(Feature f){
	double_test = Feature.FORBIDDEN_VALUE; 
	int_test = Feature.FORBIDDEN_VALUE;
	boolarray_to_string_test = null;

	name = f.name;
    }

    FeatureTest(double d, Feature f){
	this(f);

	double_test = d;
	type = Feature.CONTINUOUS;
    }

    FeatureTest(int i, Feature f){
	this(f);

	int_test = i;
	type = Feature.INTEGER;
    }

    FeatureTest(BoolArray b, Feature f){
	this(f);

	boolarray_to_string_test = b.duplicate();
	type = Feature.NOMINAL;
    }

    public String check_trivial_branching(Dataset ds, Feature f, boolean prevent_zero_measure){
	// checks whether the split induces a trivial branching on f
	// if prevent_zero_measure AND f is continuous, adds double_test == f.dmin for TRIVIAL_BRANCHING_RIGHT (prevents zero measure after eventual split)
	
	if ( (!f.type.equals(type)) || (!f.name.equals(name)) )
	    Dataset.perror("FeatureTest.class :: checking trivial branching on wrong type / name");

	String ret = null;
	
	if (Feature.IS_CONTINUOUS(type)){
	    if ( (double_test < f.dmin) || ( (prevent_zero_measure) && (double_test <= f.dmin) ) )
		ret = TRIVIAL_BRANCHING_RIGHT;
	    else if (double_test >= f.dmax)
		ret = TRIVIAL_BRANCHING_LEFT;
	    else
		ret = NO_TRIVIAL_BRANCHING;
	}else if (Feature.IS_INTEGER(type)){
	    if (int_test < f.imin)
		ret = TRIVIAL_BRANCHING_RIGHT;
	    else if (int_test >= f.imax)
		ret = TRIVIAL_BRANCHING_LEFT;
	    else
		ret = NO_TRIVIAL_BRANCHING;
	}else if (Feature.IS_NOMINAL(type)){
	    Vector <String> moda = getStringTest(ds); // get all String corresponding to the test
	    if (moda == null)
		Dataset.perror("Feature.class :: no test for nominal feature " + name);
	    boolean all_my_moda_outside_f = true, all_f_moda_in_mine = true;
	    int i;

	    for (i=0;i<moda.size();i++)
		if (f.modalities.contains(moda.elementAt(i)))
		    all_my_moda_outside_f = false;

	    for (i=0;i<f.modalities.size();i++)
		if (!moda.contains(f.modalities.elementAt(i)))
		    all_f_moda_in_mine = false;

	    if ( (all_my_moda_outside_f) && (all_f_moda_in_mine) )
		Dataset.perror("Feature.class :: inconsistency in check_trivial_branching");

	    if (all_my_moda_outside_f)
		ret = TRIVIAL_BRANCHING_RIGHT;
	    else if (all_f_moda_in_mine)
		ret = TRIVIAL_BRANCHING_LEFT;
	    else
		ret = NO_TRIVIAL_BRANCHING;
	}
	return ret;
    }

    public double [] rapid_stats_split_measure_hard(LocalEmpiricalMeasure parent_measure, Dataset ds, Feature f_split, boolean unspecified_attribute_handling_biased){
	// simplified version of split_measure_hard: just returns statistics

	double [] ret = new double[2];
	int i, index_feature_in_e = ds.indexOfFeature(name);

	for (i=0;i<parent_measure.observations_indexes.length;i++){
      if ((!Observation.FEATURE_IS_UNKNOWN(
              ds.observations_from_file.elementAt(parent_measure.observations_indexes[i]),
              index_feature_in_e))
          && (!type.equals(
              ds.observations_from_file
                  .elementAt(parent_measure.observations_indexes[i])
                  .typed_features
                  .elementAt(index_feature_in_e)
                  .type)))
        Dataset.perror(
            "FeatureTest.class :: type mismatch to split examples ( "
                + type
                + " != "
                + ds.observations_from_file
                    .elementAt(parent_measure.observations_indexes[i])
                    .typed_features
                    .elementAt(index_feature_in_e)
                    .type
                + ")");

	    if (observation_goes_left(ds.observations_from_file.elementAt(parent_measure.observations_indexes[i]), ds, f_split, unspecified_attribute_handling_biased))
		ret[0] += 1.0;
	    else
		ret[1] += 1.0;
	}
	
	return ret;
    }

    public LocalEmpiricalMeasure [] split_measure_hard(LocalEmpiricalMeasure parent_measure, Dataset ds, Feature f_split, boolean unspecified_attribute_handling_biased){
	// hard split of the measure according to the feature
	
	if (parent_measure.observations_indexes == null)
	    return null;

	int index_feature_in_e = ds.indexOfFeature(name);
	
	Vector <Integer> left_indexes = new Vector<>();
	Vector <Integer> right_indexes = new Vector<>();

	LocalEmpiricalMeasure [] ret;

	int i;
	for (i=0;i<parent_measure.observations_indexes.length;i++){
      if ((!Observation.FEATURE_IS_UNKNOWN(
              ds.observations_from_file.elementAt(parent_measure.observations_indexes[i]),
              index_feature_in_e))
          && (!type.equals(
              ds.observations_from_file
                  .elementAt(parent_measure.observations_indexes[i])
                  .typed_features
                  .elementAt(index_feature_in_e)
                  .type)))
        Dataset.perror(
            "FeatureTest.class :: type mismatch to split examples ( "
                + type
                + " != "
                + ds.observations_from_file
                    .elementAt(parent_measure.observations_indexes[i])
                    .typed_features
                    .elementAt(index_feature_in_e)
                    .type
                + ")");

	    if (observation_goes_left(ds.observations_from_file.elementAt(parent_measure.observations_indexes[i]), ds, f_split, unspecified_attribute_handling_biased))
		left_indexes.addElement(new Integer(parent_measure.observations_indexes[i]));
	    else
		right_indexes.addElement(new Integer(parent_measure.observations_indexes[i]));
	}

	ret = new LocalEmpiricalMeasure[2];
	ret[0] = new LocalEmpiricalMeasure(left_indexes.size());
	ret[1] = new LocalEmpiricalMeasure(right_indexes.size());

	int [] tab_left, tab_right;

	tab_left = new int[left_indexes.size()];
	tab_right = new int[right_indexes.size()];

	for (i=0;i<left_indexes.size();i++)
	    tab_left[i] = left_indexes.elementAt(i).intValue();
	
	for (i=0;i<right_indexes.size();i++)
	    tab_right[i] = right_indexes.elementAt(i).intValue();

    // check size sum
    if (tab_left.length + tab_right.length != parent_measure.observations_indexes.length)
      Dataset.perror(
          "FeatureTest.class :: size mismatch to split examples ( "
              + tab_left.length
              + " + "
              + tab_right.length
              + " != "
              + parent_measure.observations_indexes.length
              + ")");

	ret[0].init_indexes(tab_left);
	ret[1].init_indexes(tab_right);
	
	return ret;
    }

    public LocalEmpiricalMeasure [] split_measure_soft(LocalEmpiricalMeasure parent_measure, Dataset ds, Feature f_split){
	// soft split of the measure according to the feature
	
	if (parent_measure.observations_indexes == null)
	    return null;

	int index_feature_in_e = ds.indexOfFeature(name);

	LocalEmpiricalMeasure [] ret = new LocalEmpiricalMeasure[2];
	ret[0] = new LocalEmpiricalMeasure(0);
	ret[1] = new LocalEmpiricalMeasure(0);
	
	double [] p_loc, p_new;
	Observation oo;

	int i;
	for (i=0;i<parent_measure.observations_indexes.length;i++){
	    p_new = new double [2];
	    oo = ds.observations_from_file.elementAt(parent_measure.observations_indexes[i]);
      if ((!Observation.FEATURE_IS_UNKNOWN(oo, index_feature_in_e))
          && (!type.equals(oo.typed_features.elementAt(index_feature_in_e).type)))
        Dataset.perror(
            "FeatureTest.class :: type mismatch to split examples ( "
                + type
                + " != "
                + oo.typed_features.elementAt(index_feature_in_e).type
                + ")");

	    p_loc = share_observation_goes_left(oo, ds, f_split);
	    p_new[0] = p_loc[0] * parent_measure.proportions[i];
	    p_new[1] = p_loc[1] * parent_measure.proportions[i];

	    if (p_new[0] + p_new[1] == 0.0)
		Dataset.perror("Feature.class :: observation count split in zero probabilities");
	    
	    if (p_new[0] > 0.0)
		ret[0].add(parent_measure.observations_indexes[i], p_new[0]);

	    if (p_new[1] > 0.0)
		ret[1].add(parent_measure.observations_indexes[i], p_new[1]);
	}

	if ( (!ret[0].contains_indexes()) && (!ret[1].contains_indexes()) )
	    Dataset.perror("Feature.class :: no indexes kept from " + parent_measure);
	
	if (!ret[0].contains_indexes())
	    ret[0] = null;

	if (!ret[1].contains_indexes())
	    ret[1] = null;
	
	return ret;
    }
    
    public int [][] split_observations(int [] observation_indexes, Dataset ds, Feature f_split, boolean unspecified_attribute_handling_biased){ 
	// returns an array with two arrays: ret[0] = left observations; ret[1] = right observations
	// they must match with indexes in dataset

	if (observation_indexes == null)
	    return null;

	int index_feature_in_e = ds.indexOfFeature(name);
	
	Vector <Integer> left_indexes = new Vector<>();
	Vector <Integer> right_indexes = new Vector<>();
	int [][] ret = new int [2][];

	int i;
	for (i=0;i<observation_indexes.length;i++){
      if ((!Observation.FEATURE_IS_UNKNOWN(
              ds.observations_from_file.elementAt(observation_indexes[i]), index_feature_in_e))
          && (!type.equals(
              ds.observations_from_file
                  .elementAt(observation_indexes[i])
                  .typed_features
                  .elementAt(index_feature_in_e)
                  .type)))
        Dataset.perror(
            "FeatureTest.class :: type mismatch to split examples ( "
                + type
                + " != "
                + ds.observations_from_file
                    .elementAt(observation_indexes[i])
                    .typed_features
                    .elementAt(index_feature_in_e)
                    .type
                + ")");

	    if (observation_goes_left(ds.observations_from_file.elementAt(observation_indexes[i]), ds, f_split, unspecified_attribute_handling_biased))
		left_indexes.addElement(new Integer(observation_indexes[i]));
	    else
		right_indexes.addElement(new Integer(observation_indexes[i]));
	}

	ret[0] = new int[left_indexes.size()];
	ret[1] = new int[right_indexes.size()];

	for (i=0;i<left_indexes.size();i++)
	    ret[0][i] = left_indexes.elementAt(i).intValue();
	
	for (i=0;i<right_indexes.size();i++)
	    ret[1][i] = right_indexes.elementAt(i).intValue();

    // check size sum
    if (ret[0].length + ret[1].length != observation_indexes.length)
      Dataset.perror(
          "FeatureTest.class :: size mismatch to split examples ( "
              + ret[0].length
              + " + "
              + ret[1].length
              + " != "
              + observation_indexes.length
              + ")");

	return ret;
    }
    
    public Feature [] split_feature(Dataset ds, Feature f, boolean check_non_zero_measure, boolean check_consistency){
	// returns TWO features by applying this to f, [left, right]
	// thus, USES THE DOMAIN OF f, NOT that in ds
	
	Feature [] ft = new Feature[2];
	Feature left = null, right = null;
	Vector <String> vright;
	Vector <String> moda, modaleft, modaright;
	int i;

	if (check_consistency)
	    checkConsistency(ds, f, check_non_zero_measure); // after that, split can guarantee non-zero support measure on both features, ASSUMING f is in a Tree (not from a Dataset)

	if (Feature.IS_CONTINUOUS(f.type)){
	    left = new Feature(f.name, f.type, f.modalities, f.dmin, double_test, check_non_zero_measure);
	    right = new Feature(f.name, f.type, f.modalities, double_test, f.dmax, check_non_zero_measure);
	}else if (Feature.IS_INTEGER(f.type)){
	    
	    left = new Feature(f.name, f.type, f.modalities, f.imin, int_test); 
	    right = new Feature(f.name, f.type, f.modalities, int_test + 1, f.imax); // CHECK: was true
	}else if (Feature.IS_NOMINAL(f.type)){
	    moda = getStringTest(ds); // get all String corresponding to the test
	    if (moda == null)
		Dataset.perror("Feature.class :: no test for nominal feature " + f);

	    // compute modaleft
	    modaleft = new Vector <>();
	    modaright = new Vector <>();
	    for (i=0;i<f.modalities.size();i++)
		if (moda.contains(f.modalities.elementAt(i)))
		    modaleft.addElement(new String(f.modalities.elementAt(i)));
		else
		    modaright.addElement(new String(f.modalities.elementAt(i)));

	    if (modaleft.size() == 0)
		Dataset.perror("Feature.class :: no modality to add to the left split");
	    
	    if (modaright.size() == 0)
		Dataset.perror("Feature.class :: no modality to add to the right split");
	    
	    left = new Feature(f.name, f.type, modaleft, f.dmin, f.dmax);
	    right = new Feature(f.name, f.type, modaright, f.dmin, f.dmax);
	}

	ft[0] = left;
	ft[1] = right;

	return ft;
    }


    public double [] share_observation_goes_left(Observation e, Dataset ds, Feature f_node){
	// path followed in the tree by an observation
	// continuous OR integer values : <= is left, > is right
	// nominal values : in the set is left, otherwise is right
	// f_node = feature split by this, IN A TREE

	// returns an array[2], [0] = p_left; [1] = p_right
	// both values sum to 1.0 and can be non = 0 when unknonwn value
	
	int index_feature_in_e = ds.indexOfFeature(name);

	checkConsistency(ds, f_node, false);
	double cv;
	int ci;
	String nv;
	Vector <String> ssv;
	int i;

	double p_left;
	double [] ret = new double[2];
	boolean found = false;

	p_left = -1.0;
	if (Observation.FEATURE_IS_UNKNOWN(e, index_feature_in_e)){
	    if (Feature.IS_CONTINUOUS(type)){
        if (f_node.dmax == f_node.dmin)
          Dataset.perror("Feature.class :: f_node.dmax = " + f_node.dmax + " == f_node.dmin ");

		    p_left = ( double_test - f_node.dmin ) / (f_node.dmax - f_node.dmin);
		}else if (Feature.IS_INTEGER(type)){
        if (f_node.imax == f_node.imin)
          Dataset.perror("Feature.class :: f_node.imax = " + f_node.imax + " == f_node.imin ");

		    p_left = ( (double) int_test - f_node.imin + 1 ) / ((double) f_node.imax - f_node.imin + 1);
		}else if (Feature.IS_NOMINAL(type))
		    p_left = ((double) boolarray_to_string_test.cardinality())/((double) f_node.modalities.size()); 
		else
		    Dataset.perror("Feature.class :: no type available for feature " + this);
	}else if (Feature.IS_CONTINUOUS(type)){
      if ((e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.NOMINAL))
          || (e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.INTEGER)))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a Double");

	    cv = e.typed_features.elementAt(index_feature_in_e).dv;
	    if (cv <= double_test)
		p_left = 1.0;
	    else
		p_left = 0.0;
	}else if (Feature.IS_INTEGER(type)){
      if ((e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.NOMINAL))
          || (e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.CONTINUOUS)))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a Double");

	    ci = e.typed_features.elementAt(index_feature_in_e).iv;
	    if (ci <= int_test)
		p_left = 1.0;
	    else
		p_left = 0.0;
	}else if (Feature.IS_NOMINAL(type)){
      if ((e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.CONTINUOUS))
          || (e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.INTEGER)))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a String");

	    nv = e.typed_features.elementAt(index_feature_in_e).sv;
	    ssv = getStringTest(ds);
	    found = false;
	    i = 0;
	    do{
		if (nv.equals((String) ssv.elementAt(i)))
		    found = true;
		else
		    i++;
	    }while( (!found) && (i<ssv.size()) );
	    
	    if (found)
		p_left = 1.0;
	    else
		p_left = 0.0;
	}else
	    Dataset.perror("Feature.class :: no type / value available for feature " + this);

	if (p_left == -1.0)
	    Dataset.perror("Feature.class :: error in the computation of p_left ");

	ret[0] = p_left;
	ret[1] = 1.0 - p_left;

	return ret;
    }

    
    public boolean observation_goes_left(Observation e, Dataset ds, Feature f_node, boolean unspecified_attribute_handling_biased){ 
	// path followed in the tree by an observation
	// continuous OR integer values : <= is left, > is right
	// nominal values : in the set is left, otherwise is right

	// f_node = feature split by this, IN A TREE
	// unspecified_attribute_handling_biased = true => uses local domain and split to decide random branching, else Bernoulli(0.5)

	int index_feature_in_e = ds.indexOfFeature(name);

	checkConsistency(ds, f_node, false);
	// important: for nominal attributes, ensures the strings corresponding to this are a subset of f_node.modalities
	
	double cv;
	int ci;
	String nv;
	Vector <String> ssv;
	int i;

	double p_left;

	if (Observation.FEATURE_IS_UNKNOWN(e, index_feature_in_e)){
	    if (!unspecified_attribute_handling_biased){
		if (Statistics.RANDOM_P_NOT_HALF() < 0.5)
		    return true;
		else
		    return false;
	    }else{
		p_left = -1.0;
		if (Feature.IS_CONTINUOUS(type)){
          if (f_node.dmax == f_node.dmin)
            Dataset.perror("Feature.class :: f_node.dmax = " + f_node.dmax + " == f_node.dmin ");
		    p_left = ( double_test - f_node.dmin ) / (f_node.dmax - f_node.dmin);
		}else if (Feature.IS_INTEGER(type)){
          if (f_node.imax == f_node.imin)
            Dataset.perror("Feature.class :: f_node.imax = " + f_node.imax + " == f_node.imin ");
		    p_left = ( (double) int_test - f_node.imin + 1 ) / ((double) f_node.imax - f_node.imin + 1);
		}else if (Feature.IS_NOMINAL(type))
		    p_left = ((double) boolarray_to_string_test.cardinality())/((double) f_node.modalities.size());
		else
		    Dataset.perror("Feature.class :: no type available for feature " + this);

		if (Statistics.RANDOM_P_NOT(p_left) < p_left)
		    return true;
		else
		    return false;
	    }
	}

	if (Feature.IS_CONTINUOUS(type)){
      if ((e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.NOMINAL))
          || (e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.INTEGER)))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a Double");

	    cv = e.typed_features.elementAt(index_feature_in_e).dv;
	    if (cv <= double_test)
		return true;
	    return false;
	}else if (Feature.IS_INTEGER(type)){
      if ((e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.NOMINAL))
          || (e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.CONTINUOUS)))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a Double");

	    ci = e.typed_features.elementAt(index_feature_in_e).iv;
	    if (ci <= int_test)
		return true;
	    return false;
	}else if (Feature.IS_NOMINAL(type)){
      if ((e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.CONTINUOUS))
          || (e.typed_features.elementAt(index_feature_in_e).type.equals(Feature.INTEGER)))
        Dataset.perror(
            "Feature.class :: wrong class match : "
                + e.typed_features.elementAt(index_feature_in_e)
                + " not a String");

	    nv = e.typed_features.elementAt(index_feature_in_e).sv;
	    ssv = getStringTest(ds);

	    // tag_opt	    
	    for (i=0;i<ssv.size();i++){
		if (nv.equals((String) ssv.elementAt(i)))
		    return true;
	    }
	    return false;
	}else
	    Dataset.perror("Feature.class :: no type available for feature " + this);

	return false;
    }

    public static FeatureTest copyOf(FeatureTest fv, Feature f){

    if (!fv.name.equals(f.name))
      Dataset.perror("FeatureTest.class :: no copy possible using features of different names");

	FeatureTest fr = null;

	if (fv.type.equals(Feature.CONTINUOUS))
	    fr = new FeatureTest(fv.getDoubleTest(), f);
	else if (fv.type.equals(Feature.INTEGER))
	    fr = new FeatureTest(fv.getIntTest(), f);
	else if (fv.type.equals(Feature.NOMINAL))
	    fr = new FeatureTest(fv.getBoolArrayToStringTest(), f);
	else
	    Dataset.perror("FeatureTest.class :: no such type as " + fv.type);
	return fr;
    }

    public boolean equals(Object o){
	if (o == this)
	    return true;
	if (!(o instanceof FeatureTest))
	    return false;
	FeatureTest ft = (FeatureTest) o;

	if (!ft.type.equals(type))
	    return false;

	if ( (type.equals(Feature.NOMINAL)) && (!boolarray_to_string_test.equals(ft.getBoolArrayToStringTest())) )
	    return false;
	else if ( (type.equals(Feature.INTEGER)) && (int_test != ft.getIntTest()) )
	    return false;
	else if ( (type.equals(Feature.CONTINUOUS)) && (double_test != ft.getDoubleTest()) )
	    return false;
	
	return true;
    }
    
    public void checkConsistency(Feature f){
    if (!type.equals(f.type))
      Dataset.perror(
          "FeatureTest.class :: incompatible feature types (" + type + " vs " + f.type + ")");
	if ( (type.equals(Feature.NOMINAL)) && (f.modalities == null) )
	    Dataset.perror("FeatureTest.class :: test computed on a feature w/o modalities");
    }

    public void checkConsistency(Dataset ds, Feature f, boolean check_non_zero_measure){
    if (!type.equals(f.type))
      Dataset.perror(
          "FeatureTest.class :: incompatible feature types ("
              + type
              + " vs Feature "
              + f.type
              + ")");

    if (!name.equals(f.name))
      Dataset.perror(
          "FeatureTest.class :: incompatible feature names ("
              + name
              + " vs Feature "
              + f.name
              + ")");

	if (type.equals(Feature.NOMINAL)){
      if (f.modalities == null)
        Dataset.perror("FeatureTest.class :: empty modalities in Feature for test's consistency");
	    int i;
	    Vector<String> vs =  getStringTest(ds);

	    if (check_non_zero_measure){
		// must have a modality not in tests AND test contain at least one element
		boolean found = false;

        if (vs.size() == 0)
          Dataset.perror("FeatureTest.class :: [check_non_zero_measure] test has no String " + f);

		// checks that at least 1 modality of f is in the test (goes left)
		found = false;
		i = 0;
		do{
		    if (vs.contains(f.modalities.elementAt(i)))
			found = true;
		    else
			i++;
		}while( (!found) && (i<f.modalities.size()) );

        if (!found)
          Dataset.perror("FeatureTest.class :: all modalities of " + this + " are not in " + f);

		// checks that at least 1 modality of f is not in the test (goes right)
		found = false;
		i = 0;
		do{
		    if (!vs.contains(f.modalities.elementAt(i)))
			found = true;
		    else
			i++;
		}while( (!found) && (i<f.modalities.size()) );

        if (!found)
          Dataset.perror("FeatureTest.class :: all modalities of " + f + " are in " + this);
	    }
	}else if (type.equals(Feature.CONTINUOUS)){
      // does NOT ensure non zero measure
      if ((double_test < f.dmin) || (double_test > f.dmax))
        Dataset.perror(
            "FeatureTest.class :: test value " + double_test + " not in domain of Feature " + f);

	    if (check_non_zero_measure){
        // check double_test \in (f.dmin, f.dmax)

        if ((double_test <= f.dmin) || (double_test >= f.dmax))
          Dataset.perror(
              "FeatureTest.class :: [check_non_zero_measure] test value "
                  + double_test
                  + " not strictly in domain of Feature "
                  + f);
	    }
	}else if (type.equals(Feature.INTEGER)){
      // does NOT ensure non zero measure
      if ((int_test < f.imin) || (int_test > f.imax))
        Dataset.perror(
            "FeatureTest.class :: test value " + int_test + " not in domain of Feature " + f);

	    if (check_non_zero_measure){
        // check int_test \in {f.imin, f.min + 1, ..., f.imax - 1}

        if (int_test == f.imax)
          Dataset.perror(
              "FeatureTest.class :: [check_non_zero_measure] test value "
                  + int_test
                  + " equals the right bound of Feature "
                  + f);
	    }
	}
    }
    
    public double getDoubleTest(){
	//checkConsistency();
	if (!type.equals(Feature.CONTINUOUS))
	    Dataset.perror("FeatureTest.class :: unauthorized call for double w/ type " + type);
	return double_test;
    }
    
    public int getIntTest(){
	//checkConsistency();
	if (!type.equals(Feature.INTEGER))
	    Dataset.perror("FeatureTest.class :: unauthorized call for integer w/ type " + type);
	return int_test;
    }
    
    public BoolArray getBoolArrayToStringTest(){
	//checkConsistency();
	if (!type.equals(Feature.NOMINAL))
	    Dataset.perror("FeatureTest.class :: unauthorized call for nominal w/ type " + type);
	return boolarray_to_string_test;
    }

    public Vector<String> getStringTest(Dataset ds){
	Feature fds = ds.features.elementAt(ds.indexOfFeature(name));
	
	checkConsistency(fds);
	Vector<String> vs = new Vector<>();
    if (boolarray_to_string_test.size() != fds.modalities.size())
      Dataset.perror(
          "FeatureTest.class :: BoolArray "
              + boolarray_to_string_test
              + " of different *size* ("
              + boolarray_to_string_test.size()
              + ") than the number of modalities ("
              + fds.modalities.size()
              + ")");
	int i;
	for (i=0;i<fds.modalities.size();i++)
	    if (boolarray_to_string_test.get(i))
		vs.addElement(new String(fds.modalities.elementAt(i)));
	return vs;
    }
    
    public String toString(Dataset ds){
	String v = name;
	int i;
	Vector <String> ssv;

	if (Feature.IS_CONTINUOUS(type))
	    v += " <= " + DF6.format(double_test);
	else if (Feature.IS_INTEGER(type))
	    v += " <= " + int_test;
	else if (Feature.IS_NOMINAL(type)){
	    v += " in {";
	    ssv = getStringTest(ds);
	    for (i = 0; i < ssv.size();i++){
		v += ssv.elementAt(i);
		if (i<ssv.size() - 1)
		    v += ", ";
	    }
	    v += "}";
	}else
	    Dataset.perror("FeatureTest.class :: no type available for feature " + this);
	return v;
    }
}

/**************************************************************************************************************************************
 * Class FeatureValue
 * provides encapsulation for all types that a feature can take, incl. unknown value
 *****/

class FeatureValue implements Debuggable{
    public static String S_UNKNOWN = "-1";

    public static boolean IS_UNKNOWN(double d){
	String s = "" + d;
	return s.equals(S_UNKNOWN);
    }
    
    public static boolean IS_UNKNOWN(int i){
	String s = "" + i;
	return s.equals(S_UNKNOWN);
    }

    public static boolean IS_UNKNOWN(String s){
	return s.equals(S_UNKNOWN);
    }

    public static boolean IS_UNKNOWN(Object o){
	if (!o.getClass().getSimpleName().equals("FeatureValue"))
	    return false;

	FeatureValue v = (FeatureValue) o;
	if (v.type.equals(Feature.UNKNOWN_VALUE))
	    return true;
	
	return false;
    }

    boolean is_unknown;
    double dv;
    int iv;
    String sv;

    String type;

    FeatureValue(){
	is_unknown = true;

	type = Feature.UNKNOWN_VALUE;
	iv = Feature.FORBIDDEN_VALUE;
	dv = (double) Feature.FORBIDDEN_VALUE;
	sv = "" + Feature.FORBIDDEN_VALUE;
    }
    
    FeatureValue(String s){
	sv = s;
	is_unknown = false;
	
	type = Feature.NOMINAL;
	iv = Feature.FORBIDDEN_VALUE;
	dv = (double) Feature.FORBIDDEN_VALUE;
    }

    FeatureValue(double d){
	dv = d;
	is_unknown = false;

	type = Feature.CONTINUOUS;
	iv = Feature.FORBIDDEN_VALUE;
	sv = "" + Feature.FORBIDDEN_VALUE;
    }

    FeatureValue(int i){
	iv = i;
	is_unknown = false;

	type = Feature.INTEGER;
	dv = (double) Feature.FORBIDDEN_VALUE;
	sv = "" + Feature.FORBIDDEN_VALUE;
    }

    public String toString(){
	if (type.equals(Feature.UNKNOWN_VALUE))
	    return "[?]";
	else if (type.equals(Feature.NOMINAL))
	    return sv;
	else if (type.equals(Feature.INTEGER))
	    return "" + iv;
	else if (type.equals(Feature.CONTINUOUS))
	    return "" + dv;
	else
	    Dataset.perror("FeatureValue.class :: no such type as " + type);
	return "";
    }

    public static FeatureValue copyOf(FeatureValue fv){
	if (fv.type.equals(Feature.UNKNOWN_VALUE))
	    return new FeatureValue();
	else if (fv.type.equals(Feature.NOMINAL))
	    return new FeatureValue(fv.sv);
	else if (fv.type.equals(Feature.CONTINUOUS))
	    return new FeatureValue(fv.dv);
	else if (fv.type.equals(Feature.INTEGER))
	    return new FeatureValue(fv.iv);
	Dataset.perror("FeatureValue.class :: no value type " + fv.type);
	return new FeatureValue();
    }

    public boolean equals(Object o){
	if (o == this)
	    return true;
	if (!(o instanceof FeatureValue))
	    return false;
	FeatureValue fv = (FeatureValue) o;

	if (!fv.type.equals(type))
	    return false;

	if ( (is_unknown) && (fv.is_unknown) )
	    return true;
	else if ( (!is_unknown) && (fv.is_unknown) )
	    return false;
	else if ( (is_unknown) && (!fv.is_unknown) )
	    return false;
	
	if ( (type.equals(Feature.NOMINAL)) && (!sv.equals(fv.sv)) )
	    return false;
	else if ( (type.equals(Feature.INTEGER)) && (iv != fv.iv) )
	    return false;
	else if ( (type.equals(Feature.CONTINUOUS)) && (dv != fv.dv) )
	    return false;

	return true;
    }
}

/**************************************************************************************************************************************
 * Class Feature
 *****/

class Feature implements Debuggable{
  public static String NOMINAL = "NOMINAL",
      CONTINUOUS = "CONTINUOUS",
      INTEGER = "INTEGER",
      UNKNOWN_VALUE = "UNKNOWN_VALUE";

    public static String TYPE[] = {Feature.NOMINAL, Feature.CONTINUOUS, Feature.INTEGER};
    public static int TYPE_INDEX(String s){
	int i = 0;
	while(i<TYPE.length){
	    if (TYPE[i].equals(s))
		return i;
	    i++;
	}
	return -1;
    }
    
    public static String DISPERSION_NAME[] = {"Entropy", "Variance", "Variance"};

    // Discriminator relevant variables
    public static int NUMBER_CONTINUOUS_TIES = 100;
    // splits the interval in this number of internal splits (results in N+1 subintervals) FOR CONTINUOUS VARIABLES

    public static int FORBIDDEN_VALUE = -100000;
    // value to initialise doubles and int. Must not be in dataset
    
    public static boolean DISPLAY_TESTS = true;
    
    // All purpose variables
    String name;
    String type;

    //Restricted to domain features, to collect the info about data (useful for splits)
    double [] observed_doubles;
    int [] observed_ints;

    //Feature specific domain stuff -- redesign w/ a specific class for Generics
    //characterisation of feature's domain
    Vector <String> modalities; //applies only to Feature.NOMINAL features
    double dmin, dmax; //applies only to Feature.CONTINUOUS features
    int imin, imax;  //applies only to Feature.INTEGER features

    double dispertion_statistic_value;
    // Entropy for nominal, variance for ordered

    public static String SAVE_FEATURE(Feature f){
	String ret = "";
	ret += f.name + "\t" + f.type + "\t";
	int i;

	if (Feature.IS_NOMINAL(f.type)){
	    if (f.modalities == null)
		ret += "null";
	    else if (f.modalities.size() == 0)
		ret += "{}";
	    else
		for (i=0;i<f.modalities.size();i++){
		    ret += (String) f.modalities.elementAt(i);
		    if (i<f.modalities.size()-1)
			ret += "\t";
		}
	}else if (Feature.IS_CONTINUOUS(f.type)){
	    ret += f.dmin + "\t" + f.dmax;
	}else if (Feature.IS_INTEGER(f.type)){
	    ret += f.imin + "\t" + f.imax;
	}
	return ret;
    }

    public double length(){
	double l = -1.0;
	if (Feature.IS_CONTINUOUS(type))
	    l = dmax - dmin;
	else if (Feature.IS_INTEGER(type))
	    l = (double) imax - (double) imin + 1.0;
	else if (Feature.IS_NOMINAL(type))
	    l = (double) modalities.size();

	if (l<0.0)
	    Dataset.perror("Feature.class :: feature " + this + " has <0 length");
	
	return l;
    }

    public boolean equals(Object o){
	int i, j;
	
	if (o == this)
	    return true;
	if (!(o instanceof Feature))
	    return false;
	Feature f = (Feature) o;
	if (!(( ( (Feature.IS_NOMINAL(f.type)) && (Feature.IS_NOMINAL(type)) )
	       || ( (Feature.IS_INTEGER(f.type)) && (Feature.IS_INTEGER(type)) )
		|| ( (Feature.IS_CONTINUOUS(f.type)) && (Feature.IS_CONTINUOUS(type)) ) )))
	    return false;

	if (Feature.IS_INTEGER(f.type))
	    if ( (f.imin != imin)  || (f.imax != imax) )
		return false;

	if (Feature.IS_CONTINUOUS(f.type))
	    if ( (f.dmin != dmin)  || (f.dmax != dmax) )
		return false;

	if (Feature.IS_NOMINAL(f.type)){
	    if (f.modalities.size() != modalities.size())
		return false;
	    for (i=0;i<f.modalities.size();i++)
		if (!((String) f.modalities.elementAt(i)).equals(modalities.elementAt(i)))
		    return false;
	}

	return true;
    }
    
    public static Feature copyOf(Feature f){
	Vector <String> v = null;
	double miv = (double) Feature.FORBIDDEN_VALUE, mav = (double) Feature.FORBIDDEN_VALUE;
	
	if (Feature.IS_NOMINAL(f.type))
	    v = new  Vector<String> (f.modalities);
	else if (Feature.IS_CONTINUOUS(f.type)){
	    miv = f.dmin;
	    mav = f.dmax;
	}else if (Feature.IS_INTEGER(f.type)){
	    miv = f.imin;
	    mav = f.imax;
	}
	
	Feature fn = new Feature(f.name, f.type, v, miv, mav);
	
	if (Feature.IS_CONTINUOUS(f.type)){
	    fn.imin = fn.imax = Feature.FORBIDDEN_VALUE;
	}else if (Feature.IS_INTEGER(f.type)){
	    fn.dmin = fn.dmax = Feature.FORBIDDEN_VALUE;
	}else if (Feature.IS_NOMINAL(f.type)){
	    fn.imin = fn.imax = Feature.FORBIDDEN_VALUE;
	    fn.dmin = fn.dmax = Feature.FORBIDDEN_VALUE;
	}
	
	return fn;
    }

    public static boolean OBSERVATION_MATCHES_FEATURE(Observation e, Feature f, int f_index){
    // checks whether e.typed_features.elementAt(f_index) is in domain of f

    if (e.typed_features.elementAt(f_index).type.equals(Feature.UNKNOWN_VALUE))
      Dataset.perror("Feature.class :: forbidden unknown value for OBSERVATION_MATCHES_FEATURE");
	    //return true;

	String f_type = e.typed_features.elementAt(f_index).type;
	
	double ed;
	int ei;
	String es;
	
	if (f_type.equals(Feature.CONTINUOUS)){
	    if (!Feature.IS_CONTINUOUS(f.type))
		Dataset.perror("Feature.class :: feature type mismatch -- CONTINUOUS");
	    
	    ed = e.typed_features.elementAt(f_index).dv;
	    if ( (ed >= f.dmin) && (ed <= f.dmax) )
		return true;
	    else
		return false;
	}else if (f_type.equals(Feature.INTEGER)){
	    if (!Feature.IS_INTEGER(f.type))
		Dataset.perror("Feature.class :: feature type mismatch -- INTEGER");
	    
	    ei = e.typed_features.elementAt(f_index).iv;
	    if ( (ei >= f.imin) && (ei <= f.imax) )
		return true;
	    else
		return false;
	}else if (f_type.equals(Feature.NOMINAL)){
	    if (!Feature.IS_NOMINAL(f.type))
		Dataset.perror("Feature.class :: feature type mismatch -- NOMINAL");
	    
	    es = e.typed_features.elementAt(f_index).sv;
	    if (f.modalities.contains(es))
		return true;
	    else
		return false;
	}else
	    Dataset.perror("Feature.class :: feature type unknown");

	return true;
    }

    public static Feature CAP(Feature[] features, boolean ensure_non_zero_measure){
	// cheaper option to compute the intersection of any number of features
	// proceeds by feature; if one \cap returns null, returns null
	
	if ( (features == null) || (features.length == 0) )
	    Dataset.perror("Feature.class :: no intersection possible with 0 features");
	
	int i, j;
	String n = "", t;
	for (i=0;i<features.length-1;i++){
      if (!features[i].type.equals(features[i + 1].type))
        Dataset.perror(
            "Feature.class :: no intersection possible between features of different types");
      if (!features[i].name.equals(features[i + 1].name))
        Dataset.perror(
            "Feature.class :: no intersection possible between features of different names");
	}

	t = features[0].type;
	n = features[0].name;
	
	if (Feature.IS_CONTINUOUS(t)){
	    double cap_dmin = -1.0, cap_dmax = -1.0;
	    for (i=0;i<features.length;i++){
		if ( (i==0) || (features[i].dmin > cap_dmin) ){
		    cap_dmin = features[i].dmin;
		}
		if ( (i==0) || (features[i].dmax < cap_dmax) ){
		    cap_dmax = features[i].dmax;
		}
	    }

	    System.out.println(n + "{" + cap_dmin + "," + cap_dmax + "}");

	    if ( (cap_dmin > cap_dmax) || ( (cap_dmin == cap_dmax) && (ensure_non_zero_measure) ) )
		return null;

	    return new Feature(n, t, null, cap_dmin, cap_dmax);
	}else if (Feature.IS_INTEGER(t)){
	    int cap_imin = -1, cap_imax = -1;
	    for (i=0;i<features.length;i++){
		if ( (i==0) || (features[i].imin > cap_imin) ){
		    cap_imin = features[i].imin;
		}
		if ( (i==0) || (features[i].imax < cap_imax) ){
		    cap_imax = features[i].imax;
		}
	    }
	    if (cap_imin > cap_imax)
		return null;

	    return new Feature(n, t, null, (double) cap_imin, (double) cap_imax);
	}else if (Feature.IS_NOMINAL(t)){
	    Vector <String> cap_string = null;
	    int sinit;
	    for (i=0;i<features.length;i++){
		if (i==0)
		    cap_string = new Vector<>((Vector <String>) features[0].modalities);
		else{
		    sinit = cap_string.size(); 
		    for (j=sinit-1;j>=0;j--){
			if (!features[i].modalities.contains(cap_string.elementAt(j)))
			    cap_string.removeElementAt(j);
		    }
		}
	    }
	    if ( (cap_string == null) || (cap_string.size() == 0) )
		return null;

	    return new Feature(n, t, cap_string, -1.0, -1.0);
	}else
	    Dataset.perror("Feature.class :: no type " + t + " for feature intersection");

	return null;
    }

    public Feature cap(Feature t, boolean ensure_non_zero_measure){
	// return the intersection feature: EXPENSIVE REPLACE BY CAP
	
	if ( (!name.equals(t.name)) || (!type.equals(t.type)) )
	    Dataset.perror("Feature.class :: feature intersection between different features");

	Feature ret = Feature.copyOf(this);
	int i;
	
	if (Feature.IS_CONTINUOUS(type)){

	    if  ( ( (ensure_non_zero_measure) && ( (dmax <= t.dmin) || (dmin >= t.dmax) ) )
		  || ( (!ensure_non_zero_measure) && ( (dmax < t.dmin) || (dmin > t.dmax) ) ) )
		return  null;
	    
	    ret.dmin = (dmin >= t.dmin) ? dmin : t.dmin;
	    ret.dmax = (dmax <= t.dmax) ? dmax : t.dmax;
	}else if (Feature.IS_INTEGER(type)){
	    if ( (imax < t.imin) || (imin > t.imax) )
		return null;

	    if (imax == t.imin){
		ret.imin = ret.imax = imax;
	    }else if (imin == t.imax){
		ret.imin = ret.imax = imin;
	    }else{	    
		ret.imin = (imin >= t.imin) ? imin : t.imin;
		ret.imax = (imax <= t.imax) ? imax : t.imax;
	    }
	}else if (Feature.IS_NOMINAL(type)){

	    if ( (modalities == null) || (t.modalities == null) )
		return null;

	    Vector <String> inter = new Vector<>();

	    for (i=0;i<modalities.size();i++)
		if (t.modalities.contains(modalities.elementAt(i)))
		    inter.addElement(new String(modalities.elementAt(i)));

	    ret.modalities = inter;
	}else
	    Dataset.perror("Feature.class :: feature type unknown");
	return ret;
    }

    public static void TEST_UNION(Feature f_parent, Feature f_left, Feature f_right){
	// controls the union of the children features is the parent
	// ensures the children have non-empty domain
	int i;

    if ((!f_parent.type.equals(f_left.type)) || (!f_parent.type.equals(f_right.type)))
      Dataset.perror(
          "Feature.class :: parent feature of type "
              + f_parent.type
              + " but children: "
              + f_left.type
              + ", "
              + f_right.type);

	if (Feature.IS_CONTINUOUS(f_parent.type)){
	    if ( (f_left.dmin != f_parent.dmin) || (f_right.dmax != f_parent.dmax) )
		Dataset.perror("Feature.class :: double domain does not cover parent's range");
	    if (f_left.dmax != f_right.dmin)
		Dataset.perror("Feature.class :: double domain union mismatch");
	}else if (Feature.IS_INTEGER(f_parent.type)){
	    if ( (f_left.imin != f_parent.imin) || (f_right.imax != f_parent.imax) )
		Dataset.perror("Feature.class :: integer domain does not cover parent's range");
      if (f_left.imax + 1 != f_right.imin)
        Dataset.perror(
            "Feature.class :: integer domain union mismatch : f_left.imax = "
                + f_left.imax
                + ", f_right.imin = "
                + f_right.imin);
	}else if (Feature.IS_NOMINAL(f_parent.type)){
	    if ( (f_left.modalities == null) || (f_right.modalities == null) )
		Dataset.perror("Feature.class :: nominal domain has null domain in a child");
	    if ( (f_left.modalities.size() == 0) || (f_right.modalities.size() == 0) )
		Dataset.perror("Feature.class :: nominal domain has empty domain in a child");
	    if (f_parent.modalities == null)
		Dataset.perror("Feature.class :: nominal domain has null domain in parent");
	    if (f_parent.modalities.size() == 0)
		Dataset.perror("Feature.class :: nominal domain has empty domain in parent");
      for (i = 0; i < f_left.modalities.size(); i++)
        if (!f_parent.modalities.contains((String) f_left.modalities.elementAt(i)))
          Dataset.perror(
              "Feature.class :: parent's nominal domain does not contain left child modality "
                  + ((String) f_left.modalities.elementAt(i)));
      for (i = 0; i < f_right.modalities.size(); i++)
        if (!f_parent.modalities.contains((String) f_right.modalities.elementAt(i)))
          Dataset.perror(
              "Feature.class :: parent's nominal domain does not contain right child modality "
                  + ((String) f_right.modalities.elementAt(i)));
      if (f_left.modalities.size() + f_right.modalities.size() != f_parent.modalities.size())
        Dataset.perror(
            "Feature.class :: parent's nominal domain contains modalities not in children");
	}
    }
    
    public static boolean IS_SUBFEATURE(Feature a, Feature b){
	// checks if domain(a) \subseteq domain(b)
	return IS_SUBFEATURE(a, -1, b, -1);
    }
    
    public static boolean IS_SUBFEATURE(Feature a, int index_a, Feature b, int index_b){
	// checks if domain(a) \subseteq domain(b) AND returns an error if index_a != index_b (in myDomain.myDS.features)
	// also checks inconsistencies: one of a or b must be a subfeature of the other AND the feature type values must have been computed

	boolean anotinb, bnotina;
	int i, ia, ib;

    if (index_a != index_b)
      Dataset.perror("Feature.class :: not the same feature (" + index_a + " != " + index_b + ")");
    if (!a.type.equals(b.type))
      Dataset.perror(
          "Feature.class :: not the same type of feature (" + a.type + " != " + b.type + ")");

	if (IS_CONTINUOUS(a.type)){
	    if (a.dmin >= b.dmin){
        if (a.dmax <= b.dmax) return true;
        else
          Dataset.perror(
              "Feature.class :: inconsistency for subfeature check for : ("
                  + a.dmin
                  + ", "
                  + a.dmax
                  + ") subseteq ("
                  + b.dmin
                  + ", "
                  + b.dmax
                  + ") ? ");
      } else if (a.dmax < b.dmax)
        Dataset.perror(
            "Feature.class :: inconsistency for subfeature check for : ("
                + a.dmin
                + ", "
                + a.dmax
                + ") subseteq ("
                + b.dmin
                + ", "
                + b.dmax
                + ") ? ");
	}else if (IS_INTEGER(a.type)){	    
	    if (a.imin >= b.imin){
        if (a.imax <= b.imax) return true;
        else
          Dataset.perror(
              "Feature.class :: inconsistency for subfeature check for : ("
                  + a.imin
                  + ", "
                  + a.imax
                  + ") subseteq ("
                  + b.imin
                  + ", "
                  + b.imax
                  + ") ? ");
      } else if (a.imax < b.imax)
        Dataset.perror(
            "Feature.class :: inconsistency for subfeature check for : ("
                + a.imin
                + ", "
                + a.imax
                + ") subseteq ("
                + b.imin
                + ", "
                + b.imax
                + ") ? ");
	}else if (IS_NOMINAL(a.type)){
	    if (a.modalities == null)
		return true;
	    else if (b.modalities != null){
		anotinb = bnotina = false;
		ia = ib = -1;
		for (i=0;i<a.modalities.size();i++)
		    if (!b.modalities.contains((String) a.modalities.elementAt(i))){
			anotinb = true;
			ia = i;
		    }
		for (i=0;i<b.modalities.size();i++)
		    if (!a.modalities.contains((String) b.modalities.elementAt(i))){
			bnotina = true;
			ib = i;
		    }
        if ((anotinb) && (bnotina))
          Dataset.perror(
              "Feature.class :: inconsistency for subfeature check for : "
                  + ((String) a.modalities.elementAt(ia))
                  + " not in b and "
                  + ((String) b.modalities.elementAt(ib))
                  + " not in a ");
        else if (!anotinb) return true;
	    }
	}else
	    Dataset.perror("Feature.class :: no Feature type for " + a.type);

	return false;
    }

    public static boolean IS_CONTINUOUS(String t){
	return (t.equals(Feature.CONTINUOUS));
    }

    public static boolean IS_INTEGER(String t){
	// equiv. to Nominal Mono-valued, ordered
	
	return (t.equals(Feature.INTEGER));
    }

    static boolean IS_NOMINAL(String t){
	// Nominal Mono-Valued, no order
	
	return (t.equals(Feature.NOMINAL));
    }
    
    static int INDEX(String t){
	int i = 0;
	do {
	    if (t.equals(TYPE[i]))
		return i;
	    i++;
	}while(i < TYPE.length);
	Dataset.perror("No type found for " + t);
	return -1;
    }

    public static Feature DOMAIN_FEATURE(String n, String t, Vector <String> m, Vector <Double> vd, Vector <Integer> vi, double miv, double mav){
	Feature f = new Feature(n, t, m, miv, mav);
	int i;
	if (f.type.equals(Feature.CONTINUOUS)){
	    f.observed_doubles = new double[vd.size()];
	    for (i=0;i<vd.size();i++)
		f.observed_doubles[i] = vd.elementAt(i).doubleValue();
	    Arrays.sort(f.observed_doubles);
	}
	if (f.type.equals(Feature.INTEGER)){
	    f.observed_ints = new int[vi.size()];
	    for (i=0;i<vi.size();i++)
		f.observed_ints[i] = vi.elementAt(i).intValue();
	    Arrays.sort(f.observed_ints);
	}

	return f;
    }
    
    Feature(String n, String t, Vector <String> m, double miv, double mav){	
	name = n;
	type = t;
	modalities = null;
	observed_doubles = null;
	observed_ints = null;

    if (((Feature.IS_CONTINUOUS(t)) || (Feature.IS_INTEGER(t))) && (miv > mav))
      Dataset.perror(
          "Feature.class :: Continuous or Integer feature has min value "
              + miv
              + " > max value "
              + mav);
    else if ((Feature.IS_NOMINAL(t)) && (miv < mav))
      Dataset.perror(
          "Feature.class :: Nominal feature "
              + name
              + " has min value = "
              + miv
              + ", max value = "
              + mav
              + ", should be default Forbidden value "
              + Feature.FORBIDDEN_VALUE);

    if ((Feature.IS_CONTINUOUS(t)) && (miv >= mav))
      Dataset.perror(
          "Feature.class :: Continuous feature "
              + n
              + " has min value "
              + miv
              + " >= max value "
              + mav);

    if ((!Feature.IS_NOMINAL(t))
        && ((miv == (double) Feature.FORBIDDEN_VALUE) && (mav == (double) Feature.FORBIDDEN_VALUE)))
      Dataset.perror(
          "Feature.class :: Non nominal feature "
              + n
              + " has min value "
              + Feature.FORBIDDEN_VALUE
              + " == max value "
              + Feature.FORBIDDEN_VALUE
              + " = Forbidden value");

	if (Feature.IS_CONTINUOUS(t)){
	    dmin = miv;
	    dmax = mav;

	    imin = imax = Feature.FORBIDDEN_VALUE;
	}else if (Feature.IS_INTEGER(t)){
	    imin = (int) miv;
	    imax = (int) mav;

	    dmin = dmax = (double) Feature.FORBIDDEN_VALUE; 
	}else{
	    imin = imax = Feature.FORBIDDEN_VALUE;
	    dmin = dmax = (double) Feature.FORBIDDEN_VALUE;
	}
	
	if (Feature.IS_NOMINAL(t))
	    modalities = m;
	
	dispertion_statistic_value = -1.0;
    }

    Feature(String n, String t, Vector <String> m, double miv, double mav, boolean check_non_zero_measure){	
	name = n;
	type = t;
	modalities = null;
	observed_doubles = null;
	observed_ints = null;

    if (((Feature.IS_CONTINUOUS(t)) || (Feature.IS_INTEGER(t))) && (miv > mav))
      Dataset.perror(
          "Feature.class :: Continuous or Integer feature has min value "
              + miv
              + " > max value "
              + mav);
    else if ((Feature.IS_NOMINAL(t)) && (miv < mav))
      Dataset.perror(
          "Feature.class :: Nominal feature "
              + name
              + " has min value = "
              + miv
              + ", max value = "
              + mav
              + ", should be default Forbidden value "
              + Feature.FORBIDDEN_VALUE);

    if ((Feature.IS_CONTINUOUS(t)) && (miv >= mav) && (check_non_zero_measure))
      Dataset.perror(
          "Feature.class :: Continuous feature "
              + n
              + " has min value "
              + miv
              + " >= max value "
              + mav);

    if ((!Feature.IS_NOMINAL(t))
        && ((miv == (double) Feature.FORBIDDEN_VALUE) && (mav == (double) Feature.FORBIDDEN_VALUE)))
      Dataset.perror(
          "Feature.class :: Non nominal feature "
              + n
              + " has min value "
              + Feature.FORBIDDEN_VALUE
              + " == max value "
              + Feature.FORBIDDEN_VALUE
              + " = Forbidden value");

	if (Feature.IS_CONTINUOUS(t)){
	    dmin = miv;
	    dmax = mav;

	    imin = imax = Feature.FORBIDDEN_VALUE;
	}else if (Feature.IS_INTEGER(t)){
	    imin = (int) miv;
	    imax = (int) mav;

	    dmin = dmax = (double) Feature.FORBIDDEN_VALUE; 
	}else{
	    imin = imax = Feature.FORBIDDEN_VALUE;
	    dmin = dmax = (double) Feature.FORBIDDEN_VALUE;
	}
	
	if (Feature.IS_NOMINAL(t))
	    modalities = m;
	
	dispertion_statistic_value = -1.0;
    }
    
    // ALL PURPOSE INSTANCE METHODS
    
    public boolean has_in_range(double v){
	if ( (Feature.IS_NOMINAL(type)) || (Feature.IS_INTEGER(type)) )
	    Dataset.perror("Feature.class :: feature " + this + " queried for double value " + v);
	if (!Feature.IS_CONTINUOUS(type))
	    Dataset.perror("Feature.class :: feature type " + type + " unregistered ");
	if (v < dmin)
	    return false;
	if (v > dmax)
	    return false;
	return true;
    }

    public boolean has_in_range(int v){
	if ( (Feature.IS_NOMINAL(type)) || (Feature.IS_CONTINUOUS(type)) )
	    Dataset.perror("Feature.class :: feature " + this + " queried for double value " + v);
	if (!Feature.IS_INTEGER(type))
	    Dataset.perror("Feature.class :: feature type " + type + " unregistered ");
	if (v < imin)
	    return false;
	if (v > imax)
	    return false;
	return true;
    }

    public boolean has_in_range(String s){
    if ((Feature.IS_CONTINUOUS(type)) || (Feature.IS_INTEGER(type)))
      Dataset.perror(
          "Feature.class :: Continuous feature " + this + " queried for nominal value " + s);
	if (!Feature.IS_NOMINAL(type))
	    Dataset.perror("Feature.class :: feature type " + type + " unregistered ");

	int i;
	String ss;
	for (i=0;i<modalities.size();i++){
	    ss = (String) modalities.elementAt(i);
	    if (ss.equals(s))
		return true;
	}
	return false;
    }


    public boolean has_in_range(FeatureValue v){
    if (!type.equals(v.type))
      Dataset.perror(
          "Feature.class :: Feature "
              + this
              + " not of the same type as value "
              + v
              + " => cannot contain it");

	if ( ( (Feature.IS_NOMINAL(type)) && (has_in_range(v.sv)) )
	     ||  ( (Feature.IS_CONTINUOUS(type)) && (has_in_range(v.dv)) )
	     ||  ( (Feature.IS_INTEGER(type)) && (has_in_range(v.iv)) ) )
	    return true;

	return false;
    }
    
    public String range(boolean in_generator){
	String v = "";
	int i;
	if (Feature.IS_NOMINAL(type)){
	    v += "{";
	    for (i=0;i<modalities.size();i++){
		v += "" + modalities.elementAt(i);
		if (i<modalities.size() - 1)
		    v += ", ";
	    }
	    v += "}";
	}else if (Feature.IS_CONTINUOUS(type)){
	    v += "[" + DF4.format(dmin) + ", " + DF4.format(dmax) + "]";
	}else if (Feature.IS_INTEGER(type)){
	    if (imax == imin)
		v += "{" + imin + "}";
	    else{
		v += "{" + imin + ", " + (imin + 1) ;
		if (imax > imin + 2)
		    v += ", ...";
		if (imax > imin + 1)
		    v += ", " + imax;
		v += "}";
	    }
	}
	return v;
    }

    public String toString(){
	String v = "";
	int i;
    v +=
        name
            + " -- "
            + type
            + " in "
            + range(false)
            + " ["
            + Feature.DISPERSION_NAME[Feature.TYPE_INDEX(type)]
            + " = "
            + DF4.format(dispertion_statistic_value)
            + "]";

	return v;
    }

    public String toShortString(){
	String v = "";
	int i;
	v += "[ " + name + " : " + type + " in " + range(false) + " ]";
	    
	return v;
    }



    public String toStringInTree(boolean internalnode){
	String v = ""; 
	int i;
	if (internalnode)
	    v += name + " (" + type + ") in " + range(true) + ";";
	else
	    v += "(" + name + " in " + range(true) + ")";

	return v;
    }

}

