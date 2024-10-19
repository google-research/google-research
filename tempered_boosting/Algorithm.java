import java.io.*;
import java.util.*;
import java.text.SimpleDateFormat;

class Algorithm implements Debuggable {
  public static boolean STOP_IF_NO_IMPROVEMENT = false;
  // if true, stops when no improvement is spotted, during the weight update

  public static String CLAMPED = "CLAMPED", NOT_CLAMPED = "NOT_CLAMPED";

  Vector<Boost> all_algorithms;
  Domain myDomain;

  public static String[] MONTHS = {
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
  };

  public static int GET_SPLITTABLE_INDEX(DecisionTreeNode nn, DecisionTree dt, Vector cur_splits) {
    int r = -1, i, ibest = -1;
    double deltacur, deltabest = -1;
    boolean found = false;

    for (i = 0; i < cur_splits.size(); i++) {
      deltacur = ((Double) ((Vector) cur_splits.elementAt(i)).elementAt(0)).doubleValue();

      if ((SPLITTABLE(dt, nn, (Vector) cur_splits.elementAt(i)))
          && ((!found) || (deltacur > deltabest))) {
        deltabest = deltacur;
        ibest = i;
        found = true;
      }
    }

    r = ibest;

    return r;
  }

  public static boolean SPLITTABLE(DecisionTree t, DecisionTreeNode n, Vector v) {
    if ((((Double) v.elementAt(10)).doubleValue() == 0.0)
        || (((Double) v.elementAt(11)).doubleValue() == 0.0)
        || (((Double) v.elementAt(12)).doubleValue() == 0.0)
        || (((Double) v.elementAt(13)).doubleValue()
            == 0.0)) // all the weights wrt tempered weights
    return false;

    if ((((Vector) v.elementAt(4)).size() == 0) || (((Vector) v.elementAt(5)).size() == 0))
      return false;

    return true;
  }

  Algorithm(Domain dom) {
    all_algorithms = new Vector<>();
    myDomain = dom;
  }

  public void addAlgorithm(Vector all_params) {
    String d = (String) all_params.elementAt(0);

    String clamp = Algorithm.NOT_CLAMPED;
    if (all_params.size() == 5) clamp = (String) all_params.elementAt(4);

    double val_t;
    if (d.equals(Boost.KEY_NAME_TEMPERED_LOSS))
      val_t = Double.parseDouble((String) all_params.elementAt(3));
    else val_t = -1.0; // not the tempered loss;

    all_algorithms.addElement(
        new Boost(
            myDomain,
            d,
            Integer.parseInt((String) all_params.elementAt(1)),
            Integer.parseInt((String) all_params.elementAt(2)),
            val_t,
            clamp));
  }

  public Vector go() {
    int i;
    Vector vcur, vret = new Vector();
    for (i = 0; i < all_algorithms.size(); i++) {
      System.out.print("Algorithm #" + (i + 1) + "/" + all_algorithms.size() + " -- ");

      vcur = ((Boost) all_algorithms.elementAt(i)).boost(i + 1);
      vret.addElement(vcur);
    }
    return vret;
  }

  public void processTreeGraphs(Experiments e) {
    int[] ret, dum;
    ret = MonotonicTreeGraph.PROCESS_TREE_GRAPHS(DecisionTreeSkipTreeArc.USE_CARDINALS, this);
    dum =
        MonotonicTreeGraph.PROCESS_TREE_GRAPHS(DecisionTreeSkipTreeArc.USE_BOOSTING_WEIGHTS, this);

    if (ret != null) {
      e.index_algorithm_plot = ret[0];
      e.index_split_CV_plot = ret[1];
      e.index_tree_number_plot = ret[2];

      e.plot_ready = true;
    }
  }

  public void save(Vector v) {
    System.out.print("Saving results... ");

    String nameSave = myDomain.myDS.pathSave + "results_" + Utils.NOW + ".txt";
    int i, j, k;
    Vector vi, vj;
    double[] vv;

    FileWriter f = null;

    // saves in this order:
    //
    // COMPLETE SUBSET OF TREES: err = empirical risk on fold
    // COMPLETE SUBSET OF TREES: perr = estimated true risk on fold
    // COMPLETE SUBSET OF TREES: tree number
    // COMPLETE SUBSET OF TREES: tree total size (total card of all nodes)

    // COMPLETE SUBSET OF TREES: err = empirical risk of MonotonicTreeGraph on fold
    // COMPLETE SUBSET OF TREES: perr = estimated true risk of MonotonicTreeGraph on fold

    int smax = 6;
    int index_err_test = 1;
    int index_err_test_MTG = 5;

    String[] names = {
      "Err_Emp (%) ",
      "Err_Test (%)",
      "Tree #        ",
      "Total #nodes  ",
      "Err_E_MTG (%) ",
      "Err_T_MTG (%)"
    };

    double[] entries;
    double[] entries2;
    double[] as;

    double[] all_test_errs = new double[all_algorithms.size()];
    double[] all_test_errs2 = new double[all_algorithms.size()];
    String[] all_names = new String[all_algorithms.size()];

    double tp, dval;
    double thr = 0.1;

    try {
      f = new FileWriter(nameSave);
      f.write(
          "%Domain "
              + myDomain.myDS.domainName
              + " with classes centered wrt "
              + Dataset.FIT_CLASS_MODALITIES[Dataset.DEFAULT_INDEX_FIT_CLASS]
              + "\n");
      f.write(
          "%Proportion of sign(+/-) in classes : \t"
              + DF.format(myDomain.myDS.getProportionExamplesSign(true))
              + "/"
              + DF.format(myDomain.myDS.getProportionExamplesSign(false))
              + "\n");
      f.write("%Domain #examples : \t\t\t" + myDomain.myDS.number_examples_total + "\n");
      f.write("%Domain #features : \t\t\t" + myDomain.myDS.number_initial_features + "\n");
      f.write("%Max #cand. splits: \t\t\t" + Boost.MAX_SPLIT_TEST + "\n");
      f.write("%eta_noise (LS)   : \t\t\t" + myDomain.myDS.eta_noise + "\n");

      f.write("\n%Single algorithms statistics ::\n");
      for (i = 0; i < all_algorithms.size(); i++) {
        f.write("%Algo_" + i + " = " + ((Boost) all_algorithms.elementAt(i)).fullName() + "\n");
        vi = (Vector) v.elementAt(i);

        if (vi.size() != NUMBER_STRATIFIED_CV)
          Dataset.perror("not the right statistics in vector " + i);

        for (j = 0; j < smax; j++) {
          entries = new double[NUMBER_STRATIFIED_CV];
          for (k = 0; k < NUMBER_STRATIFIED_CV; k++)
            entries[k] = ((Double) ((Vector) vi.elementAt(k)).elementAt(j)).doubleValue();
          as = new double[2];
          Statistics.avestd(entries, as);
          f.write(names[j] + "\t" + DF.format(as[0]) + "\t\\pm\t" + DF.format(as[1]) + "\n");

          if (j == 1) {
            all_test_errs[i] = as[0];
            all_test_errs2[i] = as[0];
            all_names[i] = new String(((Boost) all_algorithms.elementAt(i)).fullName());
          }

          if ((j == smax - 1)
              && (((Boost) all_algorithms.elementAt(i)).name.equals(Boost.KEY_NAME_LOG_LOSS))) {
            entries = new double[NUMBER_STRATIFIED_CV]; // DT
            entries2 = new double[NUMBER_STRATIFIED_CV]; // MDT

            for (k = 0; k < NUMBER_STRATIFIED_CV; k++) {
              entries[k] =
                  ((Double) ((Vector) vi.elementAt(k)).elementAt(index_err_test)).doubleValue();
              entries2[k] =
                  ((Double) ((Vector) vi.elementAt(k)).elementAt(index_err_test_MTG)).doubleValue();
            }

            tp = -1.0; // replate by your statistical test for comparing entries, entries2

            f.write(
                "P-val("
                    + DF.format(all_test_errs[i])
                    + " == "
                    + DF.format(as[0])
                    + ")=\t"
                    + DF.format(tp)
                    + "\t--> ");
            if (tp < thr) f.write("REJECT");
            else f.write("KEEP");
            f.write("\n");
          }
        }
        f.write("\n");
      }

      f.write("\n% Test errs paired t-tests :: (threshold = " + thr + ") \n");
      for (i = 0; i < all_algorithms.size() - 1; i++)
        for (j = i + 1; j < all_algorithms.size(); j++) {
          f.write(
              "H0 = (%Algo_"
                  + i
                  + " = "
                  + ((Boost) all_algorithms.elementAt(i)).fullName()
                  + " == ");
          f.write("%Algo_" + j + " = " + ((Boost) all_algorithms.elementAt(j)).fullName() + ") ? ");

          vi = (Vector) v.elementAt(i);
          vj = (Vector) v.elementAt(j);

          entries = new double[NUMBER_STRATIFIED_CV];
          entries2 = new double[NUMBER_STRATIFIED_CV];

          for (k = 0; k < NUMBER_STRATIFIED_CV; k++) {
            entries[k] =
                ((Double) ((Vector) vi.elementAt(k)).elementAt(index_err_test))
                    .doubleValue(); // ERR TEST
            entries2[k] =
                ((Double) ((Vector) vj.elementAt(k)).elementAt(index_err_test))
                    .doubleValue(); // ERR TEST
          }

          tp = -1.0; // replate by your statistical test for comparing entries, entries2

          f.write("P-val=\t" + DF.format(tp) + "\t--> ");
          if (tp < thr) f.write("REJECT");
          else f.write("KEEP");
          f.write("\n");
        }

      f.write("\n% Test errs ordered from smallest\n");
      QuickSort.quicksort(all_test_errs, all_names);
      for (i = 0; i < all_algorithms.size(); i++)
        f.write(DF.format(all_test_errs[i]) + ", " + all_names[i] + "\n");

      f.close();
    } catch (IOException e) {
      Dataset.perror("LinearBoost.class :: Saving results error in file " + nameSave);
    }
    System.out.println("ok.");
  }
}
