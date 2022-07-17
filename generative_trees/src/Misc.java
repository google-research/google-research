// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.text.DecimalFormat;

interface Debuggable {

  static int ICML_SAVE_EACH = 20;
  // saves each ICML_SAVE_EACH + last iteration

  static DecimalFormat DF8 = new DecimalFormat("#0.00000000");
  static DecimalFormat DF6 = new DecimalFormat("#0.000000");
  static DecimalFormat DF4 = new DecimalFormat("#0.0000");
  static DecimalFormat DF2 = new DecimalFormat("#0.00");
  static DecimalFormat DF1 = new DecimalFormat("#0.0");

  static boolean Debug = false;

  // variables used to optimize a bit space + time
  static boolean SAVE_MEMORY = true;
  static boolean SAVE_TIME = true;

  static double EPS = 1E-4;
  static double EPS2 = 1E-6;
  static double EPS3 = 1E-10;
  static double EPS4 = 1E-20;
}
