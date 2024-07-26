// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.util.*;

public class Statistics implements Debuggable{

    public static double PRIOR = 0.5;
    
    public static Random R = new Random();

    public static double RANDOM_P_NOT_HALF(){
	double vv;
	do{
	    vv = R.nextDouble();
	}while (vv == 0.5);
	return vv;
    }

    public static double RANDOM_P_NOT(double p){
	double vv;
	do{
	    vv = R.nextDouble();
	}while (vv == p);
	return vv;
    }
    
    public static boolean APPROXIMATELY_EQUAL(double a, double b, double offset){
	if (Math.abs(a-b) > offset)
	    return false;
	return true;
    }

    // splitting criteria and the like
    // alpha-matusita

    public static double PHI_MATUSITA(double p){	
	return (2.0 * Math.sqrt(p * (1.0 - p)));
    }

    public static double PHI_ERR(double p){
	double rr;
	
	if (p <= 0.5)
	    rr = 2.0 * p;
	else
	    rr = 2.0 * (1.0 - p);
	return rr;
    }
    
    public static double PHI_MATUSITA(double alpha, double p){
    if ((alpha < 0.0) || (alpha > 1.0))
      Dataset.perror("Statistics.class PHI_MATUSITA :: alpha (" + alpha + ") should be in [0,1]");

	return ( (alpha * PHI_MATUSITA(p)) + ((1.0 - alpha) * PHI_ERR(p)) );
    }

    public static double MU_BAYES_GENERATIVE_FOREST_SIMPLE(double left_card, double right_card, double left_wud, double right_wud, int ds_card, double alpha){

	double left_weight_R = left_card / (double) ds_card;
	double right_weight_R = right_card / (double) ds_card;

	double parent_weight_R = left_weight_R + right_weight_R;
	double parent_wud = left_wud + right_wud;

	double tautrue = (PRIOR * left_weight_R + (1 - PRIOR) * left_wud) / (PRIOR * parent_weight_R + (1 - PRIOR) * parent_wud);

	double ptrue = PRIOR * left_weight_R / tautrue, pfalse = PRIOR * right_weight_R / (1.0 - tautrue);
	
	if ( (APPROXIMATELY_EQUAL(tautrue, 0.0, EPS3)) || (APPROXIMATELY_EQUAL(tautrue, 1.0, EPS3)) )
	    return PHI_MATUSITA(alpha, PRIOR);
    
	double val = tautrue * PHI_MATUSITA(alpha, ptrue) + (1.0 - tautrue) * PHI_MATUSITA(alpha, pfalse);

	return val;
    }

    public static double MU_BAYES_GENERATIVE_FOREST(MeasuredSupportAtTupleOfNodes [] msatol, double alpha){
	// computes mu Bayes as in (4) : [0] = left, [1] = right
	if ( (msatol == null) || ( (msatol[0] == null) && (msatol[1] == null) ) )
	    Dataset.perror("Statistics.class :: no mu Bayes computable, no split data available");

	Dataset ds;
	if (msatol[0] != null)
	    ds = msatol[0].geot.myDS;
	else
	    ds = msatol[1].geot.myDS;

	int left_unnormalized_card = (msatol[0] != null) ? msatol[0].generative_support.local_empirical_measure.observations_indexes.length : 0;
	int right_unnormalized_card = (msatol[1] != null) ? msatol[1].generative_support.local_empirical_measure.observations_indexes.length : 0;

	double parent_unnormalized_card = (msatol[0] != null) ? (double) msatol[0].marked_for_update_parent_msatol.generative_support.local_empirical_measure.observations_indexes.length : (double) msatol[1].marked_for_update_parent_msatol.generative_support.local_empirical_measure.observations_indexes.length;

	double left_weight_R = (double) left_unnormalized_card / (double) ds.observations_from_file.size();
	double right_weight_R = (double) right_unnormalized_card / (double) ds.observations_from_file.size();

	double parent_weight_R = (double) parent_unnormalized_card / (double) ds.observations_from_file.size();
	
	double parent_wud = (msatol[0] != null) ? (double) msatol[0].marked_for_update_parent_msatol.generative_support.support.weight_uniform_distribution : (double) msatol[1].marked_for_update_parent_msatol.generative_support.support.weight_uniform_distribution;

	double left_wud = -1.0;
	if (msatol[0] == null)
	    left_wud = parent_wud - ((double) msatol[1].generative_support.support.weight_uniform_distribution);
	else
	    left_wud = ((double) msatol[0].generative_support.support.weight_uniform_distribution);
	
	double tautrue = (PRIOR * left_weight_R + (1 - PRIOR) * left_wud) / (PRIOR * parent_weight_R + (1 - PRIOR) * parent_wud);

	double ptrue = PRIOR * left_weight_R / tautrue, pfalse = PRIOR * right_weight_R / (1.0 - tautrue);
	
	if ( (APPROXIMATELY_EQUAL(tautrue, 0.0, EPS3)) || (APPROXIMATELY_EQUAL(tautrue, 1.0, EPS3)) )
	    return PHI_MATUSITA(alpha, PRIOR);
    
	double val = tautrue * PHI_MATUSITA(alpha, ptrue) + (1.0 - tautrue) * PHI_MATUSITA(alpha, pfalse);

	return val;
    }

    public static double MU_BAYES_ENSEMBLE_OF_GENERATIVE_TREES(Dataset ds, double p_R, double p_R_left, double p_R_right, double u_left, double u_right, double alpha){

	double left_weight_R, right_weight_R, parent_weight_R, left_wud, parent_wud;
	double p_c_left_given_c, p_c_right_given_c, p_leaf;

	parent_wud = u_left + u_right;
	left_wud = u_left / parent_wud;

	double tautrue = (PRIOR * p_R_left + (1 - PRIOR) * left_wud) / (PRIOR * p_R + (1 - PRIOR) * parent_wud);

	double ptrue = PRIOR * p_R_left / tautrue, pfalse = PRIOR * p_R_right / (1.0 - tautrue);
	
	if ( (APPROXIMATELY_EQUAL(tautrue, 0.0, EPS3)) || (APPROXIMATELY_EQUAL(tautrue, 1.0, EPS3)) )
	    return PHI_MATUSITA(alpha, PRIOR);
    
	double val = tautrue * PHI_MATUSITA(alpha, ptrue) + (1.0 - tautrue) * PHI_MATUSITA(alpha, pfalse);

	return val;
    }

    // more stuff

    public static double SHANNON_ENTROPY(double [] vec){
	double v = 0.0, sum = 0.0;
	int i;
	for (i=0;i<vec.length;i++){	    
	    sum += vec[i];
	    if (vec[i] != 0.0)
		v += -vec[i] * Math.log(vec[i]);
	}
	if (!APPROXIMATELY_EQUAL(sum, 1.0, EPS2))
	    Dataset.perror("Statistics.class :: Shannon's input not on proba simplex");
	    
	return v;
    }

    
    public static void avestd(double [] data, double [] aveETstd) {
        double s,ep,ave,std;
        int j,n=data.length;
        ave=0.0;
        for (j=0;j<n;j++) ave += data[j];
        ave /= n;
        std=ep=0.0;
        for (j=0;j<n;j++) {
                s=data[j]-ave;
                ep += s;
                std += s*s;
        }
        std=Math.sqrt((std-ep*ep/n)/(n-1));
	aveETstd[0] = ave;
	aveETstd[1] = std;
    }

}
