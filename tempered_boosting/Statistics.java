import java.util.*;

public class Statistics implements Debuggable {

  public static double SQR(double x) {
    return x * x;
  }

  public static void avestd(double[] data, double[] aveETstd) {
    double s, ep, ave, std;
    int j, n = data.length;
    ave = 0.0;
    for (j = 0; j < n; j++) ave += data[j];
    ave /= n;
    std = ep = 0.0;
    for (j = 0; j < n; j++) {
      s = data[j] - ave;
      ep += s;
      std += s * s;
    }
    std = Math.sqrt((std - ep * ep / n) / (n - 1));
    aveETstd[0] = ave;
    aveETstd[1] = std;
  }

  public static boolean APPROXIMATELY_EQUAL(double a, double b, double offset) {
    if (Math.abs(a - b) > offset) return false;
    return true;
  }

  // tempered & loss stuff

  public static double CANONICAL_LINK_TEMPERED_BAYES_RISK(double tt, double p) {
    double num, den;

    if (tt == 1.0) {
      num = 2.0 * p - 1.0;
      den = Math.sqrt(p * (1.0 - p));
    } else {
      num = Math.pow(p, 2.0 - tt) - Math.pow(1.0 - p, 2.0 - tt);
      den = Math.pow(Q_MEAN(p, 1.0 - p, 1.0 - tt), 2.0 - tt);
    }

    return num / den;
  }

  public static double CANONICAL_LINK_LOG_LOSS_BAYES_RISK(double p) {
    return Math.log(p / (1.0 - p));
  }

  public static double TEMPERED_BAYES_RISK(double tt, double p) {
    if ((p == 0.0) || (p == 1.0)) return 0.0;

    return 2.0 * p * (1.0 - p) / Q_MEAN(p, 1.0 - p, 1.0 - tt);
  }

  public static double LOG_LOSS_BAYES_RISK(double p) {
    if ((p == 0.0) || (p == 1.0)) return 0.0;

    return -(p * Math.log(p)) - ((1.0 - p) * Math.log(1.0 - p));
  }

  public static double DELTA_BAYES_RISK_SPLIT(
      String loss_name,
      double tt,
      double wpos,
      double wneg,
      double wpos_left,
      double wneg_left,
      double wpos_right,
      double wneg_right) {
    double p = 0.0, q = 0.0, r = 0.0, tau, delta = -1.0, err;
    double wtot = wpos + wneg,
        wtot_left = wpos_left + wneg_left,
        wtot_right = wpos_right + wneg_right;

    if ((wtot == 0.0) || (wtot_left == 0.0) || (wtot_right == 0.0)) return 0.0;

    p = wpos_left / wtot_left;
    q = wpos / wtot;
    r = wpos_right / wtot_right;

    tau = wtot_left / wtot;

    // numerical precision control
    err = (Math.abs((wtot_left + wtot_right) - wtot)) / Math.max((wtot_left + wtot_right), wtot);
    if (err > EPS3)
      Dataset.warning(
          "Statistics.class :: wtot_left + wtot_right = "
              + (wtot_left + wtot_right)
              + " != wtot = "
              + wtot
              + " (err = "
              + err
              + ")");

    if (loss_name.equals(Boost.KEY_NAME_TEMPERED_LOSS))
      delta =
          TEMPERED_BAYES_RISK(tt, q)
              - (tau * TEMPERED_BAYES_RISK(tt, p))
              - ((1.0 - tau) * TEMPERED_BAYES_RISK(tt, r));
    else if (loss_name.equals(Boost.KEY_NAME_LOG_LOSS))
      delta =
          LOG_LOSS_BAYES_RISK(q)
              - (tau * LOG_LOSS_BAYES_RISK(p))
              - ((1.0 - tau) * LOG_LOSS_BAYES_RISK(r));
    else Dataset.perror("Statistics.class :: no loss " + loss_name);

    if (Math.abs(delta) < EPS3) delta = 0.0;

    if ((Double.isNaN(delta)) || (Double.isInfinite(delta)) || (delta < 0.0))
      Dataset.perror("Statistics.class :: Bayes risk is (" + delta + ")");

    return delta;
  }

  public static double TEMPERED_EXP(double z, double t) throws TemperedBoostException {
    double u, v;

    if (t == 1.0) v = Math.exp(z);
    else {
      u = Math.max(1.0 + ((1.0 - t) * z), 0.0);
      v = Math.pow(u, 1.0 / (1.0 - t));
    }

    if ((Double.isNaN(v)) || (Double.isInfinite(v))) {
      throw new TemperedBoostException(TemperedBoostException.INFINITE_WEIGHTS);
    }

    return v;
  }

  public static double TEMPERED_LOG(double z, double t) {
    if (((t == 1.0) && (z < 0.0)) || (z <= 0.0))
      Dataset.perror("Statistics.class :: bad argument " + z);

    double v;

    if (t == 1.0) v = Math.log(z);
    else v = ((Math.pow(z, 1.0 - t) - 1.0) / (1.0 - t));

    if ((Double.isNaN(v)) || (Double.isInfinite(v)))
      Dataset.perror("Statistics.class :: Tempered log(" + z + ") = " + v);

    return v;
  }

  public static double TEMPERED_PRODUCT(double u, double v, double t) {
    if ((t != 1.0) && ((u < 0.0) || (v < 0.0)))
      Dataset.perror("Statistics.class :: bad arguments (" + u + "," + v + ")");

    double x, w;

    if (t == 1.0) x = (u * v);
    else {
      w = Math.max(Math.pow(u, 1.0 - t) + Math.pow(v, 1.0 - t) - 1.0, 0.0);
      x = Math.pow(w, 1.0 / (1.0 - t));
    }

    if ((Double.isNaN(x)) || (Double.isInfinite(x)))
      Dataset.perror("Statistics.class :: Tempered prod " + u + " * " + v + " = " + x);

    return x;
  }

  public static double Q_MEAN(double u, double v, double q) {
    if (q == 0.0) return Math.sqrt(u * v);
    double w = (Math.pow(u, q) + Math.pow(v, q)) / 2.0;
    double x = Math.pow(w, 1.0 / q);

    if ((Double.isNaN(x)) || (Double.isInfinite(x)))
      Dataset.perror("Statistics.class :: Tempered mean " + u + " & " + v + " = " + x);

    return x;
  }

  public static double STAR(double t) {
    return 1.0 / (2.0 - t);
  }

  public static double CLAMP_CLASSIFIER(double v, double tt) {
    if (tt == 1.0) return v;

    double absval = Math.abs(1.0 / (1.0 - tt));
    if (v > absval) return absval;
    else if (v < -absval) return -absval;
    else return v;
  }

  public static double H_T(double v, double tt) {
    if (v == 0.0) return 0.0;
    double z = Math.pow(v, 1.0 - tt);
    return -z * Math.log(z);
  }
}
