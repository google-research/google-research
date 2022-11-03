// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Domain
 *****/

public class Domain implements Debuggable {
  public MemoryMonitor myMemoryMonitor;
  Dataset myDS;

  Wrapper myW;

  Domain(Wrapper w) {
    myW = w;

    myMemoryMonitor = new MemoryMonitor();

    myDS = new Dataset(this);
    myDS.load_features_and_examples();
    myDS.compute_domain_histograms();

    System.out.println(myDS);
  }

  public String memString() {
    return myMemoryMonitor.memString;
  }
}
