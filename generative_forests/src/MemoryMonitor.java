// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.awt.*;
import javax.swing.*;

public class MemoryMonitor implements Debuggable{
    public long freeMemory, totalMemory, maxMemory, memoryUsed;
    public String memString;
    public boolean stop;
    public int iter;

    public MemoryMonitor()
    {	
	memString = "";
	stop = false;
	iter = 0;
	Thread thread =
	    new Thread()
	    {
		public void run()
		{
		    while(!stop){
			compute();
			try { Thread.sleep(2000); } catch (Exception e) {}
		    }
		}
	    };
	thread.start();
    }
 
    public void stop(){
	stop = true;
    }

    public void compute()
    {
	int rr, gg, pmemu, pmemm;
	double fr;
	    
	freeMemory = Runtime.getRuntime().freeMemory();
	totalMemory = Runtime.getRuntime().totalMemory();
	maxMemory = Runtime.getRuntime().maxMemory();
	
	memoryUsed = totalMemory-freeMemory;
	
	pmemu = (int) (memoryUsed/(1024*1024));
	pmemm = (int) (maxMemory/(1024*1024));

	fr = (100.0 * (double) pmemu) / (double) pmemm;

	memString = "[" + DF2.format(fr) + "% Mem = " + (memoryUsed/(1024*1024)) + " Mb]";
    }
}
