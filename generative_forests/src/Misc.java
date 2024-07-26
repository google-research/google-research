// Companion Code to the paper "Generative Forests" by R. Nock and M. Guillame-Bert.

import java.text.DecimalFormat;

interface Debuggable {

    static String clear_string = "\033[H\033[2J";
    
    static DecimalFormat DF8 = new DecimalFormat("#0.00000000");
    static DecimalFormat DF6 = new DecimalFormat("#0.000000");
    static DecimalFormat DF4 = new DecimalFormat("#0.0000");
    static DecimalFormat DF2 = new DecimalFormat("#0.00");
    static DecimalFormat DF1 = new DecimalFormat("#0.0");

    static boolean Debug = false;
    
     //variables used to optimize a bit space + time
    static boolean SAVE_MEMORY = true;
    static boolean SAVE_TIME = true;

    static double EPS = 1E-4;
    static double EPS2 = 1E-6;
    static double EPS3 = 1E-10;
    static double EPS4 = 1E-20;    

    static double EPS8 = 1E-8;
}
