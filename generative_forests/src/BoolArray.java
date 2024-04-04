// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.io.*;
import java.util.*;

class BoolArray implements Debuggable{
    boolean [] bset;

    BoolArray(int n){
	bset = new boolean[n];
    }

    public int size(){
	return bset.length;
    }

    public BoolArray duplicate(){
	BoolArray n = new BoolArray(bset.length);
	int i;
	for (i=0;i<bset.length;i++)
	    n.bset[i] = bset[i];
	return n;
    }

    public boolean equals(Object o){
	if (o == this)
	    return true;
	if (!(o instanceof BoolArray))
	    return false;
	BoolArray ba = (BoolArray) o;

	if ( ( (bset == null) && (ba.bset != null) ) || ( (bset != null) && (ba.bset == null) ) )
	    return false;

	if (bset.length != ba.bset.length)
	    return false;

	int i;
	for (i=0;i<bset.length;i++)
	    if (bset[i] != ba.bset[i])
		return false;

	return true;
    }
    
    public boolean get(int index){
	return bset[index];
    }

    public void set(int index, boolean v){
	bset[index] = v;
    }

    public int cardinality(){
	if ( (bset == null) || (bset.length == 0) )
	    return 0;
	int i, f = 0;
	for (i=0;i<bset.length;i++)
	    if (bset[i])
		f++;
	return f;
    }

    public String toString(){
	if (bset == null)
	    return "(null)";

	String v = "{";
	int i, f = 0;
	for (i=0;i<bset.length;i++)
	    if (bset[i]){
		if (f>0)
		    v +=", ";
		v += i;
		f++;
	    }
	v += "}";
	return v;
    }
}
