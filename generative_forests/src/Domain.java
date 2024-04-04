// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

/**************************************************************************************************************************************
 * Class Domain
 *****/

public class Domain implements Debuggable{
    public MemoryMonitor myMemoryMonitor;
    Dataset myDS;

    Wrapper myW;

    Domain(Wrapper w){
	myW = w;
	
	myMemoryMonitor = new MemoryMonitor();

	myDS = new Dataset(this);
	myDS.load_features_and_observations();
	
	System.out.println(myDS);
    }
    
    public String memString(){
	return myMemoryMonitor.memString;
    }
}
