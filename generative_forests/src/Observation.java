// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Observation
 *****/

public class Observation implements Debuggable{
    // 1 observation
    
    int domain_id;

    double weight; // weights sum to 1
    
    Vector <FeatureValue> typed_features; //features with type, *including* class if any in the observed sample

    double local_density;
    // -1 is the observation is read from file (real);
    
    public static Vector <FeatureValue> TO_TYPED_FEATURES(Vector ev, Vector fv){
	Vector <FeatureValue> vv = new Vector <>();
	int i;
	Feature f;

	for (i=0;i<fv.size();i++){
	    f = (Feature) fv.elementAt(i);
	    if (FeatureValue.IS_UNKNOWN(new String((String) ev.elementAt(i))))
		vv.addElement(new FeatureValue());
	    else if (f.type.equals(Feature.CONTINUOUS)){

        if (Double.parseDouble((String) ev.elementAt(i)) == (double) Feature.FORBIDDEN_VALUE)
          Dataset.perror(
              "Observation.class :: Forbidden value "
                  + Feature.FORBIDDEN_VALUE
                  + " found in observation");

		vv.addElement(new FeatureValue(Double.parseDouble((String) ev.elementAt(i))));
	    }else if (f.type.equals(Feature.INTEGER)){

        if (Integer.parseInt((String) ev.elementAt(i)) == Feature.FORBIDDEN_VALUE)
          Dataset.perror(
              "Observation.class :: Forbidden value "
                  + Feature.FORBIDDEN_VALUE
                  + " found in observation");

		vv.addElement(new FeatureValue(Integer.parseInt((String) ev.elementAt(i))));
	    }else if (f.type.equals(Feature.NOMINAL))
		vv.addElement(new FeatureValue((String) ev.elementAt(i)));
	}

	return vv;
    }

    public static Observation copyOf(Observation e){
	// partial copy for test purpose / imputation essentially
	Observation fc = new Observation();
	int i;

	fc.domain_id = e.domain_id;
	fc.weight = e.weight;

	fc.local_density = e.local_density;

	fc.typed_features = new Vector<FeatureValue>();
	for (i=0;i<e.typed_features.size();i++){
	    if (FEATURE_IS_UNKNOWN(e, i))
		fc.typed_features.addElement(new FeatureValue());
	    else
		fc.typed_features.addElement(FeatureValue.copyOf(e.typed_features.elementAt(i)));
	}

	return fc;
    }
    
    public static boolean FEATURE_IS_UNKNOWN(Observation ee, int i){
	if ( (ee.typed_features == null) || (ee.typed_features.elementAt(i) == null) )
	    Dataset.perror("Observation.class :: no features");


	if ( (ee.typed_features.elementAt(i).type.equals(Feature.NOMINAL))
	     || (ee.typed_features.elementAt(i).type.equals(Feature.CONTINUOUS))
	     || (ee.typed_features.elementAt(i).type.equals(Feature.INTEGER)) )
	    return false;

    if (!(ee.typed_features.elementAt(i).type.equals(Feature.UNKNOWN_VALUE)))
      Dataset.perror(
          "Observation.class :: unknown feature type " + ee.typed_features.elementAt(i).type);

	return true;
    }

    Observation(){
	domain_id = -1;
	typed_features = null;
	weight = local_density = -1.0;
    }
    
    Observation(int id, Vector v, Vector fv, double w){
	domain_id = id;
	typed_features = Observation.TO_TYPED_FEATURES(v, fv);
	weight = w;
	local_density = -1.0;
    }

    Observation(int id, Vector v, Vector fv, double w, double ld){
	this(id, v, fv, w);
	local_density = ld;
    }

    public boolean contains_unknown_values(){
	int i;
	for (i=0;i<typed_features.size();i++)
	    if (FeatureValue.IS_UNKNOWN(typed_features.elementAt(i)))
		return true;
	return false;
    }

    public String toString(){
	String v = "";
	int i;
	v += "#" + domain_id + ": ";
	for (i=0;i<typed_features.size();i++)
	    v += typed_features.elementAt(i) + " ";

	v += "\n";
	return v;
    }

    public String toStringSaveDensity(int x, int y, boolean in_csv){
	String v = "";

	if (Observation.FEATURE_IS_UNKNOWN(this, x))
	    v+= FeatureValue.S_UNKNOWN;
	else
	    v+= typed_features.elementAt(x);
	if (in_csv)
	    v += Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX];
	else
	    v += Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_GNUPLOT];
	if (Observation.FEATURE_IS_UNKNOWN(this, y))
	    v+= FeatureValue.S_UNKNOWN;
	else
	    v+= typed_features.elementAt(y);
	if (in_csv)
	    v += Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX];
	else
	    v += Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_GNUPLOT];
	v += local_density;
	return v;
    }
    
    public String toStringSave(boolean in_csv){
	String v = "";
	int i;
	for (i=0;i<typed_features.size();i++){
	    if (Observation.FEATURE_IS_UNKNOWN(this, i))
		v+= FeatureValue.S_UNKNOWN;
	    else
		v+= typed_features.elementAt(i);
	    if (i<typed_features.size()-1){
		if (in_csv)
		    v += Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX];
		else
		    v += Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_GNUPLOT];
	    }
	}	    
	return v;
    }

    public int checkAndCompleteFeatures(Vector fv){
	//check that the observation has features in the domain of each feature, otherwise errs
	
	int i, vret = 0;
	Feature f;
	String fn;
	double fd;
	int id;

	for (i=0;i<fv.size();i++){
	    f = (Feature) fv.elementAt(i);
	    if (!typed_features.elementAt(i).type.equals(Feature.UNKNOWN_VALUE)){
        if (!typed_features.elementAt(i).type.equals(f.type))
          Dataset.perror(
              "Observation.class :: mismatch in types ("
                  + typed_features.elementAt(i).type
                  + " vs "
                  + f.type
                  + ") for feature "
                  + i);
		if ( (f.type.equals(Feature.CONTINUOUS)) && (!f.has_in_range(typed_features.elementAt(i).dv)) ){
          Dataset.warning(
              "Observation.class :: continuous attribute value "
                  + typed_features.elementAt(i).dv
                  + " not in range "
                  + f.range(false)
                  + " for feature "
                  + f.name);
		    vret++;
		}else if ( (f.type.equals(Feature.NOMINAL)) && (!f.has_in_range(typed_features.elementAt(i).sv)) ){
          Dataset.warning(
              "Observation.class :: continuous attribute value "
                  + typed_features.elementAt(i).sv
                  + " not in range "
                  + f.range(false)
                  + " for feature "
                  + f.name);
		    vret++;
		}else if ( (f.type.equals(Feature.INTEGER)) && (!f.has_in_range(typed_features.elementAt(i).iv)) ){
          Dataset.warning(
              "Observation.class :: continuous attribute value "
                  + typed_features.elementAt(i).iv
                  + " not in range "
                  + f.range(false)
                  + " for feature "
                  + f.name);
		    vret++;
		}
	    }
	}
	return vret;
    }
}

