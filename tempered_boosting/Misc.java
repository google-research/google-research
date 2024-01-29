import java.text.DecimalFormat;
import java.util.Random;

interface Debuggable {
  static DecimalFormat DF6 = new DecimalFormat("#0.000000");
  static DecimalFormat DF8 = new DecimalFormat("#0.00000000");
  static DecimalFormat DF = new DecimalFormat("#0.0000");
  static DecimalFormat DF0 = new DecimalFormat("#0.00");
  static DecimalFormat DF1 = new DecimalFormat("#0.0");

  static boolean Debug = false;

  // variables used to optimize a bit space + time
  static boolean SAVE_MEMORY = true;

  static double EPS = 1E-4;
  static double EPS2 = 1E-5;
  static double EPS3 = 1E-10;

  public static int NUMBER_STRATIFIED_CV = 10;

  public static double TOO_BIG_RATIO = 100.0;

  public static double INITIAL_FAN_ANGLE = Math.PI;
  public static double GRANDCHILD_FAN_RATIO = 0.8;

  public static boolean SAVE_PARAMETERS_DURING_TRAINING = true;
  public static boolean SAVE_CLASSIFIERS = true;
}
