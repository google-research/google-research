// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.text.NumberFormat;
import java.text.ParsePosition;
import java.util.*;

public class Dataset implements Debuggable{
    public static String KEY_NODES = "@NODES";
    public static String KEY_ARCS = "@ARCS";
    
    public static String PATH_GENERATIVE_MODELS = "generators";

    public static String KEY_SEPARATION_STRING[] = {"\t", ","};
    public static int SEPARATION_INDEX = 1;
    public static int SEPARATION_GNUPLOT = 0;

    public static String DEFAULT_DIR = "Datasets", SEP = "/", KEY_COMMENT = "//";

    public static String START_KEYWORD = "@";

    public static String SUFFIX_FEATURES = "features";  
    public static String SUFFIX_OBSERVATIONS = "data";  

    int number_features_total, number_observations_total_from_file, number_observations_total_generated;

    Vector <Feature> features;
    Hashtable <String, Integer> name_of_feature_to_feature;
    // keys = names, returns the Integer index of the feature with that name in features
    
    Vector <Observation> observations_from_file;
    // observations_from_file is the stored / read data

    Domain myDomain;
    String [] features_names_from_file;

    public static void perror(String error_text){
        System.out.println("\n" + error_text);
        System.out.println("\nExiting to system\n");
        System.exit(1);
    }

    public static void warning(String warning_text){
        System.out.print(" * WARNING * " + warning_text);
    }

    Dataset(Domain d){
	myDomain = d;
	number_observations_total_generated = -1;
	features_names_from_file = null;

	name_of_feature_to_feature = null;
    }

    public Feature domain_feature(int f_index){
	return features.elementAt(f_index);
    }

    public int number_domain_features(){
	return features.size();
    }

    public int indexOfFeature(String n){
	if (!name_of_feature_to_feature.containsKey(n))
	    Dataset.perror("Dataset.class :: no feature named " + n);	
	return (name_of_feature_to_feature.get(n).intValue());
    }

    public void printFeatures(){
	int i;
	System.out.println(features.size() + " features : ");
	for (i=0;i<features.size();i++)
	    System.out.println((Feature) features.elementAt(i));
    }

    public Support domain_support(){

	Support s = new Support(features, this);
    if (s.weight_uniform_distribution != 1.0)
      Dataset.perror(
          "Dataset.class :: domain support has uniform measure "
              + s.weight_uniform_distribution
              + " != 1.0");

	return s;
    }

    public void load_features_and_observations(){
	if (myDomain.myW == null)
	    Dataset.perror("Dataset.class :: use not authorised without Wrapper");
	
	FileReader e;
	BufferedReader br;
	StringTokenizer t;

	Vector <Vector<String>> observations_read = new Vector<>();
	Vector <String> current_observation;
		String [] features_types = null;
	
	String dum, n, ty;
	int i, j;

	boolean feature_names_ok = false;

	Vector <Vector<String>> observations_test = null;
	
	// features
	features = new Vector<>();
        name_of_feature_to_feature = new Hashtable <>();

	System.out.print("\nLoading features & observations data... ");

	number_features_total = -1;
	try{
	    e = new FileReader(myDomain.myW.path_and_name_of_domain_dataset);
	    br = new BufferedReader(e);

	    while ( (dum=br.readLine()) != null){
		if ( (dum.length()>1) && (!dum.substring(0,KEY_COMMENT.length()).equals(KEY_COMMENT)) ){
		    if (!feature_names_ok){
			// the first usable line must be feature names
			t = new StringTokenizer(dum,KEY_SEPARATION_STRING[SEPARATION_INDEX]);
			features_names_from_file = new String[t.countTokens()];

			i = 0;
			while(t.hasMoreTokens()){
			    n = t.nextToken();
              // must be String
              if (is_number(n))
                Dataset.perror(
                    "Dataset.class :: String "
                        + n
                        + " identified as Number; should not be the case for feature names");
			    features_names_from_file[i] = n;
			    i++;
			}

			number_features_total = features_names_from_file.length;
			feature_names_ok = true;
		    }else{
			// records all following values; checks sizes comply
			current_observation = new Vector<>();
			t = new StringTokenizer(dum,KEY_SEPARATION_STRING[SEPARATION_INDEX]);
            if (t.countTokens() != number_features_total)
              Dataset.perror(
                  "Dataset.class :: Observation string + "
                      + dum
                      + " does not have "
                      + number_features_total
                      + " features");
			while(t.hasMoreTokens())
			    current_observation.addElement(t.nextToken());
			observations_read.addElement(current_observation);
		    }
		}
	    }
	}catch(IOException eee){
      System.out.println(
          "Problem loading "
              + myDomain.myW.path_and_name_of_domain_dataset
              + " file --- Check the access to file");
	    System.exit(0);
	}

	if (myDomain.myW.density_estimation){
	    boolean feature_names_line_test = false;
	    observations_test = new Vector<>(); // used to complete the features domains if we are to do density estimation
	    
	    try{
		e = new FileReader(myDomain.myW.path_and_name_of_test_dataset);
		br = new BufferedReader(e);
		
		while ( (dum=br.readLine()) != null){
		    if ( (dum.length()>1) && (!dum.substring(0,Dataset.KEY_COMMENT.length()).equals(Dataset.KEY_COMMENT)) ){
			if (!feature_names_line_test)
			    feature_names_line_test = true; // first line must also be features
			else{
			    // records all following values; checks sizes comply
			    current_observation = new Vector<>();
			    t = new StringTokenizer(dum,Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
              if (t.countTokens() != number_features_total)
                Dataset.perror(
                    "Wrapper.class :: Observation string + "
                        + dum
                        + " does not have "
                        + number_features_total
                        + " features in *test file*");
			    while(t.hasMoreTokens())
				current_observation.addElement(t.nextToken());
			    
			    observations_test.addElement(current_observation);
			}
		    }
		}
	    }catch(IOException eee){
        System.out.println(
            "Problem loading "
                + myDomain.myW.path_and_name_of_test_dataset
                + " file --- Check the access to file");
		System.exit(0);
	    }
	}

	// checking, if density plot requested, that names are found in columns and computes myW.index_x_name , index_y_name
	if (myDomain.myW.plot_labels_names != null){
	    i = j = 0;
	    boolean found;
	    myDomain.myW.plot_labels_indexes = new int[myDomain.myW.plot_labels_names.length];
	    for (i=0;i<myDomain.myW.plot_labels_names.length;i++){
		found = false;
		j=0;
		do{
		    if (features_names_from_file[j].equals(myDomain.myW.plot_labels_names[i]))
			found = true;
		    else
			j++;
		}while( (!found) && (j<features_names_from_file.length) );
        if (!found)
          Dataset.perror(
              "Dataset.class :: feature name "
                  + myDomain.myW.plot_labels_names[i]
                  + " not found in domain features");
		myDomain.myW.plot_labels_indexes[i] = j;
	    }
	}

    System.out.print(
        "ok (#"
            + number_features_total
            + " features total, #"
            + observations_read.size()
            + " observations total)... Computing features... ");
	features_types = new String[number_features_total];
	boolean not_a_number, not_an_integer, not_a_binary;
	int idum, nvalf;
	double ddum;
	double miv, mav;
	Vector <String> vs;
	Vector <Double> vd;
	Vector <Integer> vi;
	
	// compute types
	for (i=0;i<number_features_total;i++){
	    
	    // String = nominal ? 1: text
	    not_a_number = false;
	    j = 0;
	    do{
		n = observations_read.elementAt(j).elementAt(i);
		if (!n.equals(FeatureValue.S_UNKNOWN)){
		    if (!is_number(n)){
			not_a_number = true;
		    }else
			j++;
		}else
		    j++;
	    }while( (!not_a_number) && (j<observations_read.size()) );

	    // String = nominal ? 2: binary
	    if ( (!not_a_number) && (myDomain.myW.force_binary_coding) ){
		j = 0;
		not_a_binary = false;
		do{
		    n = observations_read.elementAt(j).elementAt(i);
		    if (!n.equals(FeatureValue.S_UNKNOWN)){
			not_an_integer = false;
			idum = -1;
			try{
			    idum = Integer.parseInt(n);
			}catch(NumberFormatException nfe){
			    not_an_integer = true;
			}
		    
			if ( (not_an_integer) || ( (idum != 0) && (idum != 1) ) ){
			    not_a_binary = true;
			}else
			    j++;
		    }else
			j++;
		}while( (!not_a_binary) && (j<observations_read.size()) );
		if (!not_a_binary)
		    not_a_number = true;
	    }

	    if (not_a_number)
		features_types[i] = Feature.NOMINAL;
	    else if (!myDomain.myW.force_integer_coding)
		features_types[i] = Feature.CONTINUOUS;
	    else{
		// makes distinction integer / real
		not_an_integer = false;
		j = 0;
		do{
		    n = observations_read.elementAt(j).elementAt(i);
		    if (!n.equals(FeatureValue.S_UNKNOWN)){
			try{
			    idum = Integer.parseInt(n);
			}catch(NumberFormatException nfe){
			    not_an_integer = true;
			}
			if (!not_an_integer)
			    j++;
		    }else
			j++;
		}while( (!not_an_integer) && (j<observations_read.size()) );
		if (not_an_integer)
		    features_types[i] = Feature.CONTINUOUS;
		else
		    features_types[i] = Feature.INTEGER;
	    }
	}

	System.out.print("Types found: [");
	for (i=0;i<number_features_total;i++){
	    System.out.print(features_types[i]);
	    if (i<number_features_total-1)
		System.out.print(",");
	}
	System.out.print("] ");
	
	// compute features
	boolean value_seen;
	for (i=0;i<number_features_total;i++){
	    value_seen = false;
	    miv = mav = -1.0;
	    vs = null;
	    vd = null;
	    vi = null;
	    if (Feature.IS_NOMINAL(features_types[i])){
		vs = new Vector<>();
		for (j=0;j<observations_read.size();j++)
		    if ( (!observations_read.elementAt(j).elementAt(i).equals(FeatureValue.S_UNKNOWN)) && (!vs.contains(observations_read.elementAt(j).elementAt(i))) )
			vs.addElement(observations_read.elementAt(j).elementAt(i));

		if (myDomain.myW.density_estimation)
		    for (j=0;j<observations_test.size();j++)
			if ( (!observations_test.elementAt(j).elementAt(i).equals(FeatureValue.S_UNKNOWN)) && (!vs.contains(observations_test.elementAt(j).elementAt(i))) )
			    vs.addElement(observations_test.elementAt(j).elementAt(i));
		
		features.addElement(Feature.DOMAIN_FEATURE(features_names_from_file[i], features_types[i], vs, vd, vi, miv, mav));
	    }else{
		if (Feature.IS_CONTINUOUS(features_types[i]))
		    vd = new Vector<>();
		else if (Feature.IS_INTEGER(features_types[i]))
		    vi = new Vector<>();
		
		Vector <String> dummyv = new Vector <>();
		for (j=0;j<observations_read.size();j++)
		    if ( (!observations_read.elementAt(j).elementAt(i).equals(FeatureValue.S_UNKNOWN)) && (!dummyv.contains(observations_read.elementAt(j).elementAt(i))) )
			dummyv.addElement(observations_read.elementAt(j).elementAt(i));

		if (myDomain.myW.density_estimation)
		    for (j=0;j<observations_test.size();j++)
			if ( (!observations_test.elementAt(j).elementAt(i).equals(FeatureValue.S_UNKNOWN)) && (!dummyv.contains(observations_test.elementAt(j).elementAt(i))) )
			    dummyv.addElement(observations_test.elementAt(j).elementAt(i));
		
		nvalf = 0;
		for (j=0;j<observations_read.size();j++){
		    n = observations_read.elementAt(j).elementAt(i);
		    if (!n.equals(FeatureValue.S_UNKNOWN)){
			nvalf++;
			ddum = Double.parseDouble(n);
			if (!value_seen){
			    miv = mav = ddum;
			    value_seen = true;
			}else{
			    if (ddum < miv)
				miv = ddum;
			    if (ddum > mav)
				mav = ddum;
			}
		    }
		}

		if (myDomain.myW.density_estimation)
		    for (j=0;j<observations_test.size();j++){
			n = observations_test.elementAt(j).elementAt(i);
			if (!n.equals(FeatureValue.S_UNKNOWN)){
			    nvalf++;
			    ddum = Double.parseDouble(n);
			    if (!value_seen){
				miv = mav = ddum;
				value_seen = true;
			    }else{
				if (ddum < miv)
				    miv = ddum;
				if (ddum > mav)
				    mav = ddum;
			    }
			}
		    }

		for (j=0;j<dummyv.size();j++){
		    if (Feature.IS_CONTINUOUS(features_types[i]))
			vd.addElement(new Double(Double.parseDouble(dummyv.elementAt(j))));
		    else if (Feature.IS_CONTINUOUS(features_types[i]))
			vi.addElement(new Integer(Integer.parseInt(dummyv.elementAt(j))));
		}

        if (nvalf == 0)
          Dataset.perror(
              "Dataset.class :: feature "
                  + features_names_from_file[i]
                  + " has only unknown values");
		features.addElement(Feature.DOMAIN_FEATURE(features_names_from_file[i], features_types[i], vs, vd, vi, miv, mav));
	    }
	}
	for (i=0;i<number_features_total;i++)
	    name_of_feature_to_feature.put(features.elementAt(i).name, new Integer(i));
	
	System.out.print("ok... Computing observations... ");
	observations_from_file = new Vector<>();

	Observation ee;
	number_observations_total_from_file = observations_read.size();
	
	for (j=0;j<observations_read.size();j++){
	    ee = new Observation(j, observations_read.elementAt(j), features, 1.0 / (double) observations_read.size());
	    observations_from_file.addElement(ee);
	    if (ee.contains_unknown_values())
		myDomain.myW.has_missing_values = true;
	}

	int errfound = 0;
	for (i=0;i<number_observations_total_from_file;i++){
	    ee = (Observation) observations_from_file.elementAt(i);
	    errfound += ee.checkAndCompleteFeatures(features);
	}
    if (errfound > 0)
      Dataset.perror(
          "Dataset.class :: found "
              + errfound
              + " errs for feature domains in observations. Please correct domains in .features"
              + " file. ");

	compute_feature_statistics();

	System.out.println("ok... ");
    }

    public boolean is_number(String n){
	double test;
	boolean is_double = true;
	try{
	    test = Double.parseDouble(n);
	}catch(NumberFormatException nfe){
	    is_double = false;
	}
	if (is_double)
	    return true;
	
	ParsePosition p = new ParsePosition(0);
	NumberFormat.getNumberInstance().parse(n, p);
	return (n.length() == p.getIndex());
    }

    public void compute_feature_statistics(){
	int i, j;
	double [] probab = null;
	double expect_X2 = 0.0, expect_squared = 0.0, vex = -1.0, tot;
	double nfeat = 0.0, nnonz = 0.0;
	for (i=0;i<features.size();i++){
	    nfeat = 0.0;
	    if (Feature.IS_NOMINAL(((Feature) features.elementAt(i)).type))
		probab = new double[((Feature) features.elementAt(i)).modalities.size()];
	    else{
		expect_X2 = expect_squared = 0.0;
	    }
		
	    for (j=0;j<number_observations_total_from_file;j++)
		if (!Observation.FEATURE_IS_UNKNOWN((Observation) observations_from_file.elementAt(j), i))
		    if (Feature.IS_NOMINAL(((Feature) features.elementAt(i)).type))
			probab [((Feature) features.elementAt(i)).modalities.indexOf(observations_from_file.elementAt(j).typed_features.elementAt(i).sv)] += 1.0;
		    else{
            if (Feature.IS_INTEGER(((Feature) features.elementAt(i)).type))
              vex = (double) observations_from_file.elementAt(j).typed_features.elementAt(i).iv;
            else if (Feature.IS_CONTINUOUS(((Feature) features.elementAt(i)).type))
              vex = observations_from_file.elementAt(j).typed_features.elementAt(i).dv;
            else
              Dataset.perror(
                  "Dataset.class :: no feature type " + ((Feature) features.elementAt(i)).type);

			expect_squared += vex;
			expect_X2 += (vex * vex);
			nnonz += 1.0;
		    }

	    if (Feature.IS_NOMINAL(((Feature) features.elementAt(i)).type)){
		tot = 0.0;
		for (j=0;j<probab.length;j++)
		    tot += probab[j];
		for (j=0;j<probab.length;j++)
		    probab[j] /= tot;
		((Feature) features.elementAt(i)).dispertion_statistic_value = Statistics.SHANNON_ENTROPY(probab);
	    }else{
		if (nnonz == 0.0)
		    Dataset.perror("Dataset.class :: feature " + i + " has only unknown values");
		
		expect_X2 /= nnonz;
		expect_squared /= nnonz;
		expect_squared *= expect_squared;
		((Feature) features.elementAt(i)).dispertion_statistic_value = expect_X2 - expect_squared;
	    }
	}
    }
    
    public String toString(){
	int i, nm = observations_from_file.size();
	if (observations_from_file.size() > 10)
	    nm = 10;

	String v = "\n* Features (Unknown value coded as " + FeatureValue.S_UNKNOWN + ") --\n";
	for (i=0;i<features.size();i++)
	    v += ((Feature) features.elementAt(i)).toString() + "\n";
	v += "\n* First observations --\n";
	for (i=0;i<nm;i++)
	    v += ((Observation) observations_from_file.elementAt(i)).toString();

	return v;
    }

    public void sample_check(){
	if (observations_from_file == null)
	    Dataset.perror("Dataset.class :: no sample available");
    }

    public int [] all_indexes(int exception){
	int [] v;
	int i, ind = 0;
	if (exception == -1)
	    v = new int [observations_from_file.size()];
	else
	    v = new int [observations_from_file.size()-1];
	for (i=0;i<observations_from_file.size();i++){
	    if ( (exception == -1) || (i != exception) ){
		v[ind] = i;
		ind++;
	    }
	}
	return v;
    }
    
    public int size(){
	if (observations_from_file == null)
	    Dataset.perror("Dataset.class :: no observations");
	return observations_from_file.size();
    }
}
