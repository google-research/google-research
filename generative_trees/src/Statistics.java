// Companion Code to the paper "Generative Trees: Adversarial and Copycat" by R. Nock and M.
// Guillame-Bert, in ICML'22

import java.util.*;

public class Statistics implements Debuggable {

  public static boolean APPROXIMATELY_EQUAL(double a, double b, double offset) {
    if (Math.abs(a - b) > offset) return false;
    return true;
  }

  // splitting criteria and the like
  // alpha-matusita

  public static double PHI_MATUSITA(double p) {
    return (2.0 * Math.sqrt(p * (1.0 - p)));
  }

  public static double PHI_ERR(double p) {
    double rr;

    if (p <= 0.5) rr = 2.0 * p;
    else rr = 2.0 * (1.0 - p);
    return rr;
  }

  public static double PHI_MATUSITA(double alpha, double p) {
    if ((alpha < 0.0) || (alpha > 1.0))
      Dataset.perror("Statistics.class PHI_MATUSITA :: alpha (" + alpha + ") should be in [0,1]");

    return ((alpha * PHI_MATUSITA(p)) + ((1.0 - alpha) * PHI_ERR(p)));
  }

  public static double CANONICAL_LINK_MATUSITA(double alpha, double p) {
    if ((alpha < 0.0) || (alpha > 1.0))
      Dataset.perror(
          "Statistics.class CANONICAL_LINK_MATUSITA :: alpha (" + alpha + ") should be in [0,1]");

    if (p == 0.5) return 0.0;
    double vp;
    if (p < EPS) vp = EPS;
    else if (p > 1.0 - EPS) vp = 1.0 - EPS;
    else vp = p;

    double ret;
    if (vp < 0.5) ret = -2.0 * (1 - alpha);
    else ret = 2.0 * (1 - alpha);

    ret += alpha * (2.0 * vp - 1.0) / (Math.sqrt(vp * (1.0 - vp)));
    return ret;
  }

  public static double DELTA_PHI_SPLIT(
      double alpha,
      double wpos,
      double wneg,
      double wpos_left,
      double wneg_left,
      double wpos_right,
      double wneg_right,
      boolean relative) {
    // if relative = true, computes the relative (local) decrease of the CBR, therefore in [0,1] and
    // can be 1
    // otherwise computes the tree-wide difference, usually much smaller

    if ((alpha < 0.0) || (alpha > 1.0))
      Dataset.perror(
          "Statistics.class DELTA_PHI_SPLIT :: alpha (" + alpha + ") should be in [0,1]");

    // notations MOSTLY from Kearns and Mansour 1996
    // simplified
    double p = 0.0, q = 0.0, r = 0.0, taup, taur, delta, err, factor;
    double wtot = wpos + wneg,
        wtot_left = wpos_left + wneg_left,
        wtot_right = wpos_right + wneg_right;

    if ((wtot == 0.0) || (wtot_left == 0.0) || (wtot_right == 0.0)) return 0.0;

    p = wpos_left / wtot_left;
    q = wpos / wtot;
    r = wpos_right / wtot_right;

    if (relative) {
      taup = wtot_left / wtot;
      taur = 1.0 - taup;
      factor = 1.0;
    } else {
      taup = wtot_left;
      taur = wtot_right;
      factor = wtot;
    }

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

    delta =
        (factor * PHI_MATUSITA(alpha, q))
            - (taup * PHI_MATUSITA(alpha, p))
            - (taur * PHI_MATUSITA(alpha, r));
    if (Math.abs(delta) < EPS3) delta = 0.0;

    return delta;
  }

  // GENERATOR related stuff

  public static double SHANNON_ENTROPY(double[] vec) {
    double v = 0.0, sum = 0.0;
    int i;
    for (i = 0; i < vec.length; i++) {
      sum += vec[i];
      if (vec[i] != 0.0) v += -vec[i] * Math.log(vec[i]);
    }
    if (!APPROXIMATELY_EQUAL(sum, 1.0, EPS2))
      Dataset.perror("Statistics.class :: Shannon's input not on proba simplex");

    return v;
  }
}
