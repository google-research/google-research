import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Domain
 *****/

public class Domain implements Debuggable {
  public MemoryMonitor myMemoryMonitor;

  Dataset myDS;

  Domain(String nameD, String nameP, double eta, boolean check_labels) {

    if (!check_labels) System.out.println("Warning :: no checking of labels in stratified sample");

    myMemoryMonitor = new MemoryMonitor();

    myDS = new Dataset(nameD, nameP, this, eta);
    myDS.load_features();
    myDS.load_examples();
    myDS.generate_stratified_sample_with_check(check_labels);
  }

  public String memString() {
    return myMemoryMonitor.memString;
  }
}
