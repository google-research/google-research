import java.io.*;
import java.util.*;
import javax.swing.*;

class Experiments implements Debuggable {
  public static String KEY_HELP = "--help", KEY_RESOURCE = "-R";

  Algorithm myAlgos;
  Domain myDomain;

  JDecisionTreeViewer myViewer;

  int index_algorithm_plot, index_split_CV_plot, index_tree_number_plot;
  boolean plot_ready = false;

  public static String help() {
    String ret = "";
    ret += KEY_HELP + " : example command line\n\n";
    ret += KEY_RESOURCE + " :: name of resource file to parse algorithms\n";

    return ret;
  }

  Experiments() {
    index_algorithm_plot = index_split_CV_plot = index_tree_number_plot = -1;
  }

  public static void main(String[] arg) {
    Utils.INIT();

    int i;
    String kR = "";
    for (i = 0; i < arg.length; i++) {
      if (arg[i].equals(KEY_HELP)) {
        Dataset.perror(help());
      }

      if (arg[i].equals(KEY_RESOURCE)) kR = arg[i + 1];
    }

    if (kR.equals(new String(""))) Dataset.perror("No resource file name found in command line");

    System.out.println(
        "** Tempered Boosting with Trees -- "
            + History.CURRENT_HISTORY()
            + "\n"
            + "** Code provided without any warranty, use at your own risk\n"
            + "** See README.txt\n"
            + "** Feedback: richard.m.nock@gmail.com");

    if (SAVE_MEMORY) System.out.println("** Saving memory for processing");
    System.out.print("** Parsing resource file " + kR + " ...");

    Experiments ee = new Experiments();
    ee.go(kR);
  }

  public void go(String rs) {
    Vector v;
    parse(rs);

    myViewer = new JDecisionTreeViewer("Hyperbolic Embedding of DTs -- viewer");

    v = myAlgos.go();

    myViewer.go(this);
    myAlgos.processTreeGraphs(this);

    myAlgos.save(v);

    System.out.println("Ok...");
    myDomain.myMemoryMonitor.stop();
  }

  public void parse(String rs) {
    FileReader e;
    BufferedReader br;
    StringTokenizer t;
    String dum, n, nameD = "", nameP = "";
    Vector v;
    double eta = 0.0;
    boolean check_labels = true;

    myDomain = null;

    // Domain
    try {
      e = new FileReader(rs);
      br = new BufferedReader(e);

      while ((dum = br.readLine()) != null) {
        if ((dum.length() == 1)
            || ((dum.length() > 1)
                && (!dum.substring(0, Dataset.KEY_COMMENT.length()).equals(Dataset.KEY_COMMENT)))) {
          t = new StringTokenizer(dum, Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
          n = t.nextToken();
          if (n.equals(Dataset.KEY_DIRECTORY)) nameD = t.nextToken();
          else if (n.equals(Dataset.KEY_PREFIX)) nameP = t.nextToken();
          else if (n.equals(Dataset.KEY_NOISE)) eta = Double.parseDouble(t.nextToken());
          else if (n.equals(Dataset.KEY_CHECK_STRATIFIED_LABELS))
            check_labels = Boolean.parseBoolean(t.nextToken());
        }
      }
      e.close();
    } catch (IOException eee) {
      System.out.println("Problem loading ." + rs + " resource file --- Check the access to file");
      System.exit(0);
    }

    if (nameD.equals(new String(""))) Dataset.perror("No domain in resource file");
    if (nameP.equals(new String(""))) Dataset.perror("No prefix in resource file");

    System.out.println("\n\nDomain * " + nameP + " * in directory * " + nameD + " *\n");

    myDomain = new Domain(nameD, nameP, eta, check_labels);
    myAlgos = new Algorithm(myDomain);

    // Algos

    System.out.println("Running algorithms...\n");

    try {
      e = new FileReader(rs);
      br = new BufferedReader(e);

      while ((dum = br.readLine()) != null) {
        if ((dum.length() == 1)
            || ((dum.length() > 1)
                && (!dum.substring(0, Dataset.KEY_COMMENT.length()).equals(Dataset.KEY_COMMENT)))) {
          t = new StringTokenizer(dum, Dataset.KEY_SEPARATION_STRING[Dataset.SEPARATION_INDEX]);
          n = t.nextToken();
          if (n.equals(Dataset.KEY_ALGORITHM)) {
            v = new Vector();
            while (t.hasMoreTokens()) v.addElement(new String(t.nextToken()));
            myAlgos.addAlgorithm(v);
          }
        }
      }
      e.close();
    } catch (IOException eee) {
      System.out.println("Problem loading ." + rs + " resource file --- Check the access to file");
      System.exit(0);
    }
  }
}
