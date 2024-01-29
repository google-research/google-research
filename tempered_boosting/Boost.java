import java.io.*;
import java.util.*;

public class Boost implements Debuggable {
  public static int ITER_STOP;

  public static String KEY_NAME_TEMPERED_LOSS = "@TemperedLoss", KEY_NAME_LOG_LOSS = "@LogLoss";

  public static String[] KEY_NAME = {KEY_NAME_TEMPERED_LOSS, KEY_NAME_LOG_LOSS};
  public static String[] KEY_NAME_DISPLAY = {"TemperedLoss", "LogLoss"};

  public static double MAX_PRED_VALUE = 100.0;
  public static int MAX_SPLIT_TEST = 2000;

  public static double COEFF_GRAD = 0.001;
  public static double START_T = 0.9;

  public static String METHOD_NAME(String s) {
    int i = 0;
    do {
      if (KEY_NAME[i].equals(s)) return KEY_NAME_DISPLAY[i];
      i++;
    } while (i < KEY_NAME.length);
    Dataset.perror("Boost.class :: no keyword " + s);
    return "";
  }

  public static void CHECK_NAME(String s) {
    int i = 0;
    do {
      if (KEY_NAME[i].equals(s)) return;
      i++;
    } while (i < KEY_NAME.length);
    Dataset.perror("Boost.class :: no keyword " + s);
  }

  Domain myDomain;

  int max_number_tree, max_size_tree;
  double average_number_leaves, average_depth, tempered_t, next_t, grad_Z;

  boolean adaptive_t = false;

  String name, clamping;

  DecisionTree[] allTrees;
  DecisionTree[][] recordAllTrees;

  double[] allLeveragingCoefficients;

  MonotonicTreeGraph[][] recordAllMonotonicTreeGraphs_boosting_weights;
  MonotonicTreeGraph[][] recordAllMonotonicTreeGraphs_cardinals;

  double[] z_tilde;

  Boost(Domain d, String nn, int maxnt, int maxst, double tt, String clamped) {
    myDomain = d;
    name = nn;
    clamping = clamped;
    Boost.CHECK_NAME(name);

    max_number_tree = maxnt;
    max_size_tree = maxst;

    if (tt == -1.0) {
      adaptive_t = true;
      tempered_t = START_T;
    } else tempered_t = tt;

    if (!name.equals(Boost.KEY_NAME_TEMPERED_LOSS)) tempered_t = 1.0; // not the tempered loss

    allTrees = null;
    recordAllMonotonicTreeGraphs_boosting_weights = recordAllMonotonicTreeGraphs_cardinals = null;
    allLeveragingCoefficients = z_tilde = null;
    average_number_leaves = average_depth = 0.0;

    recordAllTrees = new DecisionTree[NUMBER_STRATIFIED_CV][];
    recordAllMonotonicTreeGraphs_boosting_weights = new MonotonicTreeGraph[NUMBER_STRATIFIED_CV][];
    recordAllMonotonicTreeGraphs_cardinals = new MonotonicTreeGraph[NUMBER_STRATIFIED_CV][];

    grad_Z = -1.0;
  }

  public String fullName() {
    String ret = Boost.METHOD_NAME(name);
    ret +=
        "["
            + max_number_tree
            + "("
            + max_size_tree
            + "):"
            + ((clamping.equals(Algorithm.CLAMPED)) ? "1" : "0")
            + "]";

    if (!adaptive_t) ret += "{" + tempered_t + "}";
    else ret += "{" + -1.0 + "}";
    return ret;
  }

  public Vector boost(int index_algo) {
    Vector v = new Vector(), v_cur = null;

    Vector<DecisionTree> all_trees;
    Vector<Double> all_leveraging_coefficients;
    Vector<Double> all_zs;

    Vector<Double> sequence_empirical_risks = new Vector<>();
    Vector<Double> sequence_true_risks = new Vector<>();
    Vector<Double> sequence_empirical_risks_MonotonicTreeGraph = new Vector<>();
    Vector<Double> sequence_true_risks_MonotonicTreeGraph = new Vector<>();
    Vector<Double> sequence_min_codensity = new Vector<>();
    Vector<Double> sequence_max_codensity = new Vector<>();
    // contain the errors on the tree set BUILT UP TO THE INDEX
    Vector<Integer> sequence_cardinal_nodes = new Vector<>();

    Vector<Vector<Double>> sequence_sequence_empirical_risks = new Vector<>();
    Vector<Vector<Double>> sequence_sequence_true_risks = new Vector<>();
    Vector<Vector<Double>> sequence_sequence_empirical_risks_MonotonicTreeGraph = new Vector<>();
    Vector<Vector<Double>> sequence_sequence_true_risks_MonotonicTreeGraph = new Vector<>();
    Vector<Vector<Double>> sequence_sequence_min_codensity = new Vector<>();
    Vector<Vector<Double>> sequence_sequence_max_codensity = new Vector<>();

    DecisionTree dt;

    int i, j, curcard = 0;
    double leveraging_mu, leveraging_alpha, expected_edge, curerr, opterr, ser_mtg, str_mtg;

    double err_fin,
        err_fin_MonotonicTreeGraph,
        err_best,
        perr_fin,
        perr_fin_MonotonicTreeGraph,
        perr_best;
    int card_nodes_fin, card_nodes_best, trees_fin, trees_best;

    double[] min_codensity = new double[max_number_tree];
    double[] max_codensity = new double[max_number_tree];

    System.out.println(fullName() + " (eta = " + myDomain.myDS.eta_noise + ")");
    for (i = 0; i < NUMBER_STRATIFIED_CV; i++) {
      all_trees = new Vector<>();
      all_leveraging_coefficients = new Vector<>();
      all_zs = new Vector<>();

      recordAllTrees[i] = new DecisionTree[max_number_tree];

      if (adaptive_t) tempered_t = Boost.START_T;

      TemperedBoostException.RESET_COUNTS();

      allTrees = null;
      allLeveragingCoefficients = null;

      System.out.print("> Fold " + (i + 1) + "/" + NUMBER_STRATIFIED_CV + " -- ");

      myDomain.myDS.init_weights(name, i, tempered_t);
      v_cur = new Vector();
      // saves in this order:
      //
      // COMPLETE SUBSET OF TREES: err = empirical risk on fold
      // COMPLETE SUBSET OF TREES: perr = estimated true risk on fold
      // COMPLETE SUBSET OF TREES: tree number
      // COMPLETE SUBSET OF TREES: tree size (total card of all nodes)

      sequence_empirical_risks = new Vector();
      sequence_true_risks = new Vector();
      sequence_empirical_risks_MonotonicTreeGraph = new Vector();
      sequence_true_risks_MonotonicTreeGraph = new Vector();
      sequence_min_codensity = new Vector();
      sequence_max_codensity = new Vector();
      sequence_cardinal_nodes = new Vector();

      min_codensity = new double[max_number_tree];

      for (j = 0; j < max_number_tree; j++) {
        System.out.print(".");
        v_cur = new Vector();

        dt = oneTree(j, i);

        leveraging_mu = dt.leveraging_mu(); // leveraging coefficient

        curcard += dt.number_nodes;
        average_number_leaves += (double) dt.leaves.size();
        average_depth += (double) dt.depth;

        all_trees.addElement(dt);

        leveraging_alpha = dt.leveraging_alpha(leveraging_mu, all_zs);

        all_leveraging_coefficients.addElement(new Double(leveraging_alpha));

        if ((SAVE_PARAMETERS_DURING_TRAINING) || (j == max_number_tree - 1)) {
          sequence_empirical_risks.addElement(
              new Double(
                  ensemble_error_noise_free(
                      all_trees, all_leveraging_coefficients, true, i, false)));
          sequence_true_risks.addElement(
              new Double(
                  ensemble_error_noise_free(
                      all_trees, all_leveraging_coefficients, false, i, false)));

          ser_mtg =
              ensemble_error_noise_free(all_trees, all_leveraging_coefficients, true, i, true);
          sequence_empirical_risks_MonotonicTreeGraph.addElement(new Double(ser_mtg));

          str_mtg =
              ensemble_error_noise_free(all_trees, all_leveraging_coefficients, false, i, true);
          sequence_true_risks_MonotonicTreeGraph.addElement(new Double(str_mtg));

          sequence_cardinal_nodes.addElement(new Integer(curcard));
        }

        if ((adaptive_t) && (j > 0))
          tempered_t = next_t; // change here otherwise inconsistencies in computations

        if (name.equals(Boost.KEY_NAME_TEMPERED_LOSS)) {
          try {
            reweight_examples_tempered_loss(
                dt, leveraging_mu, i, all_zs, j, min_codensity, max_codensity);
          } catch (TemperedBoostException eee) {
            min_codensity[j] = -1.0;
            max_codensity[j] = -1.0;
            reweight_examples_infinite_weight(dt, leveraging_mu, i, all_zs);
          }
        } else if (name.equals(Boost.KEY_NAME_LOG_LOSS)) {
          reweight_examples_log_loss(dt, leveraging_mu, i);
        } else Dataset.perror("Boost.class :: no such loss as " + name);

        if ((SAVE_PARAMETERS_DURING_TRAINING) || (j == max_number_tree - 1)) {
          sequence_min_codensity.addElement(new Double(min_codensity[j]));
          sequence_max_codensity.addElement(new Double(max_codensity[j]));
        }

        if (j % 10 == 0) System.out.print(myDomain.memString());
      }

      if (SAVE_PARAMETERS_DURING_TRAINING) {
        sequence_sequence_empirical_risks.addElement(sequence_empirical_risks);
        sequence_sequence_true_risks.addElement(sequence_true_risks);
        sequence_sequence_empirical_risks_MonotonicTreeGraph.addElement(
            sequence_empirical_risks_MonotonicTreeGraph);
        sequence_sequence_true_risks_MonotonicTreeGraph.addElement(
            sequence_true_risks_MonotonicTreeGraph);
        sequence_sequence_min_codensity.addElement(sequence_min_codensity);
        sequence_sequence_max_codensity.addElement(sequence_max_codensity);
      }

      allTrees = new DecisionTree[max_number_tree];
      allLeveragingCoefficients = new double[max_number_tree];
      for (j = 0; j < max_number_tree; j++) {
        allTrees[j] = (DecisionTree) all_trees.elementAt(j);
        recordAllTrees[i][j] = allTrees[j];
        allLeveragingCoefficients[j] =
            ((Double) all_leveraging_coefficients.elementAt(j)).doubleValue();
      }

      if (SAVE_CLASSIFIERS) save(i);

      err_fin = (Double) sequence_empirical_risks.elementAt(sequence_empirical_risks.size() - 1);
      perr_fin = (Double) sequence_true_risks.elementAt(sequence_true_risks.size() - 1);
      card_nodes_fin =
          (Integer) sequence_cardinal_nodes.elementAt(sequence_cardinal_nodes.size() - 1);
      trees_fin = max_number_tree;

      err_fin_MonotonicTreeGraph =
          (Double)
              sequence_empirical_risks_MonotonicTreeGraph.elementAt(
                  sequence_empirical_risks_MonotonicTreeGraph.size() - 1);
      perr_fin_MonotonicTreeGraph =
          (Double)
              sequence_true_risks_MonotonicTreeGraph.elementAt(
                  sequence_true_risks_MonotonicTreeGraph.size() - 1);

      v_cur.addElement(new Double(err_fin));
      v_cur.addElement(new Double(perr_fin));
      v_cur.addElement(new Double((double) trees_fin));
      v_cur.addElement(new Double((double) card_nodes_fin));

      v_cur.addElement(new Double(err_fin_MonotonicTreeGraph));
      v_cur.addElement(new Double(perr_fin_MonotonicTreeGraph));

      v.addElement(v_cur);

      System.out.print(
          "ok. \t(e-err t-err #nodes) = ("
              + DF.format(err_fin)
              + " "
              + DF.format(perr_fin)
              + " "
              + ((int) card_nodes_fin)
              + " -- "
              + DF.format(err_fin_MonotonicTreeGraph)
              + " "
              + DF.format(perr_fin_MonotonicTreeGraph)
              + ")");

      System.out.println(" (" + TemperedBoostException.STATUS() + ")");
    }

    if (SAVE_PARAMETERS_DURING_TRAINING) {
      double[] avg_sequence_empirical_risks = new double[max_number_tree];
      double[] stddev_sequence_empirical_risks = new double[max_number_tree];
      double[] avg_sequence_true_risks = new double[max_number_tree];
      double[] stddev_sequence_true_risks = new double[max_number_tree];
      double[] avg_sequence_empirical_risks_MonotonicTreeGraph = new double[max_number_tree];
      double[] stddev_sequence_empirical_risks_MonotonicTreeGraph = new double[max_number_tree];
      double[] avg_sequence_true_risks_MonotonicTreeGraph = new double[max_number_tree];
      double[] stddev_sequence_true_risks_MonotonicTreeGraph = new double[max_number_tree];
      double[] avg_sequence_min_codensity = new double[max_number_tree];
      double[] stddev_sequence_min_codensity = new double[max_number_tree];
      double[] avg_sequence_max_codensity = new double[max_number_tree];
      double[] stddev_sequence_max_codensity = new double[max_number_tree];
      double[] dumseq;
      double[] avestd = new double[2];

      for (j = 0; j < max_number_tree; j++) {
        dumseq = new double[NUMBER_STRATIFIED_CV];
        for (i = 0; i < NUMBER_STRATIFIED_CV; i++)
          dumseq[i] = sequence_sequence_empirical_risks.elementAt(i).elementAt(j).doubleValue();
        Statistics.avestd(dumseq, avestd);
        avg_sequence_empirical_risks[j] = avestd[0];
        stddev_sequence_empirical_risks[j] = avestd[1];

        dumseq = new double[NUMBER_STRATIFIED_CV];
        for (i = 0; i < NUMBER_STRATIFIED_CV; i++)
          dumseq[i] = sequence_sequence_true_risks.elementAt(i).elementAt(j).doubleValue();
        Statistics.avestd(dumseq, avestd);
        avg_sequence_true_risks[j] = avestd[0];
        stddev_sequence_true_risks[j] = avestd[1];

        dumseq = new double[NUMBER_STRATIFIED_CV];
        for (i = 0; i < NUMBER_STRATIFIED_CV; i++)
          dumseq[i] =
              sequence_sequence_empirical_risks_MonotonicTreeGraph
                  .elementAt(i)
                  .elementAt(j)
                  .doubleValue();
        Statistics.avestd(dumseq, avestd);
        avg_sequence_empirical_risks_MonotonicTreeGraph[j] = avestd[0];
        stddev_sequence_empirical_risks_MonotonicTreeGraph[j] = avestd[1];

        dumseq = new double[NUMBER_STRATIFIED_CV];
        for (i = 0; i < NUMBER_STRATIFIED_CV; i++)
          dumseq[i] =
              sequence_sequence_true_risks_MonotonicTreeGraph
                  .elementAt(i)
                  .elementAt(j)
                  .doubleValue();
        Statistics.avestd(dumseq, avestd);
        avg_sequence_true_risks_MonotonicTreeGraph[j] = avestd[0];
        stddev_sequence_true_risks_MonotonicTreeGraph[j] = avestd[1];

        dumseq = new double[NUMBER_STRATIFIED_CV];
        for (i = 0; i < NUMBER_STRATIFIED_CV; i++)
          dumseq[i] = sequence_sequence_min_codensity.elementAt(i).elementAt(j).doubleValue();
        Statistics.avestd(dumseq, avestd);
        avg_sequence_min_codensity[j] = avestd[0];
        stddev_sequence_min_codensity[j] = avestd[1];

        dumseq = new double[NUMBER_STRATIFIED_CV];
        for (i = 0; i < NUMBER_STRATIFIED_CV; i++)
          dumseq[i] = sequence_sequence_max_codensity.elementAt(i).elementAt(j).doubleValue();
        Statistics.avestd(dumseq, avestd);
        avg_sequence_max_codensity[j] = avestd[0];
        stddev_sequence_max_codensity[j] = avestd[1];
      }

      save(
          avg_sequence_empirical_risks,
          stddev_sequence_empirical_risks,
          avg_sequence_true_risks,
          stddev_sequence_true_risks,
          avg_sequence_empirical_risks_MonotonicTreeGraph,
          stddev_sequence_empirical_risks_MonotonicTreeGraph,
          avg_sequence_true_risks_MonotonicTreeGraph,
          stddev_sequence_true_risks_MonotonicTreeGraph,
          avg_sequence_min_codensity,
          stddev_sequence_min_codensity,
          avg_sequence_max_codensity,
          stddev_sequence_max_codensity,
          index_algo - 1);
    }

    System.out.println("");

    return v;
  }

  public void save(
      double[] ae,
      double[] se,
      double[] at,
      double[] st,
      double[] ae_MonotonicTreeGraph,
      double[] se_MonotonicTreeGraph,
      double[] at_MonotonicTreeGraph,
      double[] st_MonotonicTreeGraph,
      double[] amincod,
      double[] smincod,
      double[] amaxcod,
      double[] smaxcod,
      int index_algo) {
    String nameSave =
        myDomain.myDS.pathSave + "results_" + Utils.NOW + "_Algo" + index_algo + ".txt";
    int i;
    FileWriter f;

    try {
      f = new FileWriter(nameSave);

      f.write(
          "#Iter\tE_em_a\tE_em_s\tE_te_a\tE_te_s\tMDT_e_a\tMDT_e_s\tMDT_t_a\tMDT_t_s\tMinc_a"
              + "\tMinc_s\tMaxc_a\tMaxc_s\n");

      for (i = 0; i < ae.length; i++)
        f.write(
            i
                + "\t"
                + DF.format(ae[i])
                + "\t"
                + DF.format(se[i])
                + "\t"
                + DF.format(at[i])
                + "\t"
                + DF.format(st[i])
                + "\t"
                + DF.format(ae_MonotonicTreeGraph[i])
                + "\t"
                + DF.format(se_MonotonicTreeGraph[i])
                + "\t"
                + DF.format(at_MonotonicTreeGraph[i])
                + "\t"
                + DF.format(st_MonotonicTreeGraph[i])
                + "\t"
                + DF8.format(amincod[i])
                + "\t"
                + DF8.format(smincod[i])
                + "\t"
                + DF8.format(amaxcod[i])
                + "\t"
                + DF8.format(smaxcod[i])
                + "\n");

      f.close();
    } catch (IOException e) {
      Dataset.perror("Boost.class :: Saving results error in file " + nameSave);
    }
  }

  public void save(int split_CV) {
    System.out.print(" {Saving classifier... ");

    String nameSave = myDomain.myDS.pathSave + "classifiers_" + Utils.NOW + ".txt";
    FileWriter f = null;

    try {
      f = new FileWriter(nameSave, true);
      f.write(
          "=====> "
              + fullName()
              + " -- Fold "
              + (split_CV + 1)
              + " / "
              + NUMBER_STRATIFIED_CV
              + ": "
              + classifierToString());
      f.close();
    } catch (IOException e) {
      Dataset.perror("LinearBoost.class :: Saving results error in file " + nameSave);
    }

    System.out.print("ok.} ");
  }

  public String classifierToString() {
    String v = "H = ";
    int i;
    for (i = 0; i < max_number_tree; i++) {
      v += "(" + DF.format(allLeveragingCoefficients[i]) + " * T#" + i + ")";
      if (i < max_number_tree - 1) v += " + ";
    }
    v += " (" + clamping + "), where\n\n";
    for (i = 0; i < max_number_tree; i++) {
      v += "T#" + i + " = " + allTrees[i].toString();
      v += "\n";
    }
    v += "\n";
    return v;
  }

  public DecisionTree oneTree(int iter, int split_CV) {
    DecisionTree dumTree;
    dumTree = new DecisionTree(iter, this, max_size_tree, split_CV);
    dumTree.init(tempered_t);
    dumTree.grow_heavy_first();

    return dumTree;
  }

  // TEMPERED VERSION for reweighting
  public void reweight_examples_infinite_weight(
      DecisionTree dt, double mu, int split_CV, Vector<Double> all_zs) {
    // triggered if infinite weights => restricts support to infinite weights

    int i, ne = myDomain.myDS.train_size(split_CV), nzw = 0;
    double zz, ww, dumw, totsize = 0.0, newweight;
    Example ee;
    Vector<Integer> indexes_infinite = new Vector<>();

    double[] last_weights = new double[ne];
    double gz = 0.0;

    for (i = 0; i < ne; i++) {
      ee = myDomain.myDS.train_example(split_CV, i);
      ww = ee.current_boosting_weight; // tempered weight
      last_weights[i] = ww;

      try {
        dumw =
            Statistics.TEMPERED_PRODUCT(
                ww,
                Statistics.TEMPERED_EXP(
                    -mu * dt.output_boosting(ee) * ee.noisy_normalized_class, tempered_t),
                tempered_t);
        // Use the noisy class, for training (if no noise, just the regular class)
      } catch (TemperedBoostException eee) {
        indexes_infinite.addElement(new Integer(i));
        totsize += 1.0;
        TemperedBoostException.ADD(TemperedBoostException.INFINITE_WEIGHTS);
      }
    }

    newweight = 1.0 / Math.pow(totsize, Statistics.STAR(tempered_t));

    for (i = 0; i < ne; i++) {
      ee = myDomain.myDS.train_example(split_CV, i);
      if (indexes_infinite.contains(new Integer(i))) ee.current_boosting_weight = newweight;
      else ee.current_boosting_weight = 0.0;

      gz +=
          ee.current_boosting_weight
              * ((2.0 - tempered_t) * Statistics.H_T(last_weights[i], tempered_t)
                  - Statistics.H_T(ee.current_boosting_weight, tempered_t));
    }

    gz /= ((1.0 - tempered_t) * (1.0 - tempered_t));
    grad_Z = gz;

    if (adaptive_t) next_t = Math.max(0.0, Math.min(1.0, tempered_t - (grad_Z * Boost.COEFF_GRAD)));

    all_zs.addElement(new Double(1.0));
  }

  public void reweight_examples_log_loss(DecisionTree dt, double mu, int split_CV) {
    int i, ne = myDomain.myDS.train_size(split_CV);
    double ww, den;
    Example ee;

    for (i = 0; i < ne; i++) {
      ee = myDomain.myDS.train_example(split_CV, i);
      ww = ee.current_boosting_weight;

      den = ww + ((1.0 - ww) * Math.exp(mu * dt.unweighted_edge_training(ee)));

      ee.current_boosting_weight = ww / den;

      if ((ee.current_boosting_weight <= 0.0) || (ee.current_boosting_weight >= 1.0))
        Dataset.perror(
            "Boost.class :: example " + ee + "has weight = " + ee.current_boosting_weight);
    }
  }

  public void reweight_examples_tempered_loss(
      DecisionTree dt,
      double mu,
      int split_CV,
      Vector<Double> all_zs,
      int j,
      double[] min_codensity,
      double[] max_codensity)
      throws TemperedBoostException {
    int i, ne = myDomain.myDS.train_size(split_CV), nzw = 0;
    double zz, ww, dumw, z_j = 0.0, factor, minw = -1.0, expt, mindens = -1.0, maxdens = -1.0, dens;
    Example ee;
    boolean found = false;

    double[] last_weights = new double[ne];

    for (i = 0; i < ne; i++) {
      ee = myDomain.myDS.train_example(split_CV, i);
      ww = ee.current_boosting_weight; // tempered weight
      last_weights[i] = ww;

      expt = Statistics.TEMPERED_EXP(-mu * dt.unweighted_edge_training(ee), tempered_t);

      dumw = Statistics.TEMPERED_PRODUCT(ww, expt, tempered_t);
      // Use the noisy class, for training (if no noise, just the regular class)

      if (tempered_t == 1.0) z_j += dumw;
      else z_j += Math.pow(dumw, 2.0 - tempered_t);

      ee.current_boosting_weight = dumw;

      if (dumw == 0) {
        nzw++;
        TemperedBoostException.ADD(TemperedBoostException.ZERO_WEIGHTS);
      }

      if ((ee.current_boosting_weight != 0.0)
          && ((!found) || (ee.current_boosting_weight < minw))) {
        minw = ee.current_boosting_weight;
        found = true;
      }
    }

    if ((tempered_t == 1.0) && (nzw == 0)) {
      // some zero weights for AdaBoost, replace them with minimal !=0 weight

      z_j = 0.0;
      for (i = 0; i < ne; i++) {
        ee = myDomain.myDS.train_example(split_CV, i);
        if (ee.current_boosting_weight == 0.0) {
          ee.current_boosting_weight = minw;
          TemperedBoostException.ADD(TemperedBoostException.ZERO_WEIGHTS);
        }
        z_j += ee.current_boosting_weight;
      }
    }

    if (z_j == 0.0) Dataset.perror("Boost.class :: no >0 tempered weight");

    if (tempered_t != 1.0) z_j = Math.pow(z_j, Statistics.STAR(tempered_t));

    all_zs.addElement(new Double(z_j));

    double gz = 0.0;
    double pnext2, pprev2, pprev1;

    for (i = 0; i < ne; i++) {
      ee = myDomain.myDS.train_example(split_CV, i);
      ee.current_boosting_weight /= z_j;

      pprev1 = Math.pow(last_weights[i], 1.0 - tempered_t);
      pprev2 = Math.pow(last_weights[i], 2.0 - tempered_t);
      pnext2 = Math.pow(ee.current_boosting_weight, 2.0 - tempered_t);

      gz += (pnext2 * Math.log(pnext2)) / (2.0 - tempered_t);
      gz -= ee.current_boosting_weight * pprev1 * Math.log(pprev2);

      if ((TemperedBoostException.MIN_WEIGHT == -1.0)
          || (ee.current_boosting_weight < TemperedBoostException.MIN_WEIGHT))
        TemperedBoostException.MIN_WEIGHT = ee.current_boosting_weight;

      dens = Math.pow(ee.current_boosting_weight, 2.0 - tempered_t);
      if ((i == 0) || (dens < mindens)) mindens = dens;
      if ((i == 0) || (dens > maxdens)) maxdens = dens;

      if (Double.isNaN(ee.current_boosting_weight))
        Dataset.perror("Example " + i + " has NaN weight");
    }

    gz /= Math.abs(1.0 - tempered_t);
    grad_Z = gz;

    if (adaptive_t) {
      next_t = Math.max(0.0, Math.min(1.0, tempered_t - (grad_Z * Boost.COEFF_GRAD)));
      System.out.print("[" + DF.format(tempered_t) + "]");
    }

    min_codensity[j] = mindens;
    max_codensity[j] = maxdens;
  }

  public double ensemble_error_noise_free(
      Vector all_trees,
      Vector all_leveraging_coefficients,
      boolean onTraining,
      int split_CV,
      boolean use_MonotonicTreeGraph) {
    // uses the true label for all computations

    if ((all_trees == null) || (all_trees.size() == 0))
      Dataset.perror("Boost.class :: no trees to compute the error");

    Example ee;
    DecisionTree tt;
    double sumerr = 0.0, output, coeff, sum, sumtree = 0.0, totedge, sum_weights = 0.0;
    int i, j, ne;
    if (onTraining) ne = myDomain.myDS.train_size(split_CV);
    else ne = myDomain.myDS.test_size(split_CV);

    if (ne == 0) Dataset.perror("DecisionTree.class :: zero sample size to compute the error");

    for (i = 0; i < ne; i++) {
      sumtree = 0.0;
      if (onTraining) ee = myDomain.myDS.train_example(split_CV, i);
      else ee = myDomain.myDS.test_example(split_CV, i);

      if (onTraining) sum_weights += ee.current_boosting_weight;

      for (j = 0; j < all_trees.size(); j++) {
        tt = (DecisionTree) all_trees.elementAt(j);
        if (use_MonotonicTreeGraph) output = tt.output_boosting_MonotonicTreeGraph(ee);
        else output = tt.output_boosting(ee);
        coeff = ((Double) all_leveraging_coefficients.elementAt(j)).doubleValue();
        sumtree += (coeff * output);
        if (clamping.equals(Algorithm.CLAMPED))
          sumtree = Statistics.CLAMP_CLASSIFIER(sumtree, tempered_t);
      }

      if (sumtree == 0.0) {
        // random guess
        if (Utils.RANDOM_P_NOT_HALF() < 0.5) sumtree = -1.0;
        else sumtree = 1.0;
      }

      if (ee.normalized_class == 0.0)
        Dataset.perror("Boost.class :: Example " + ee + " has zero class");

      totedge = sumtree * ee.normalized_class;

      if (totedge < 0.0) sumerr += 1.0;
    }
    sumerr /= (double) ne;

    return sumerr;
  }
}
