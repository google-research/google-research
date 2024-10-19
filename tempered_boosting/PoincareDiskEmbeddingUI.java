import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.GradientPaint;
import java.awt.BasicStroke;

import javax.swing.JComponent;
import javax.swing.ImageIcon;
import javax.swing.plaf.PanelUI;

import java.util.*;
import java.text.*;
import java.awt.Font;
import java.awt.geom.Arc2D;

public class PoincareDiskEmbeddingUI extends PanelUI implements Debuggable {
  PoincareDiskEmbedding boss;
  boolean wait, tree_changed;

  static final Color[] COLORS = {Color.black, Color.white};
  static final Color[] LABEL_COLORS = {Color.green, Color.red};

  public static Color COLOR_VERTEX = Color.black;
  public static Color COLOR_EDGE = Color.darkGray;

  public static Color LABEL_FONT_COLOR = Color.blue;
  public static Color DISPLAY_FONT_COLOR = Color.black;

  public static Color ISO_CONFIDENCE_LINES_COLOR = Color.black;

  public static Color LEVERAGING_COEFFICIENTS_COLOR = Color.pink;

  static final int DISPLAY_FONT_SIZE = 20;
  static final int LABEL_FONT_SIZE = 20;
  static final Font DISPLAY_FONT = new Font("Helvetica", Font.ITALIC, DISPLAY_FONT_SIZE);
  static final Font LABEL_FONT = new Font("Helvetica", Font.PLAIN, LABEL_FONT_SIZE);
  static final Font LABEL_FONT_HIGHLIGHT = new Font("Helvetica", Font.PLAIN, LABEL_FONT_SIZE);

  public static int VERTEX_RADIUS = 4;

  public static int ALPHA_TYPE_PLOT = 0;

  public static boolean SHOW_LABELS_WITH_COLORS = true;

  public static String[] ISO_CONFIDENCE_LINES_P_TEXT = {
    "0.4|0.6", "0.3|0.7", "0.2|0.8", "0.1|0.9"
  }; // covers 1-p by symmetry; |r| = |2p-1|
  public static String[] ISO_CONFIDENCE_LINES_ALPHA_TEXT = {"1.0", "2.0", "3.0", "4.0"}; // |alpha|

  public static double[] ISO_CONFIDENCE_LINES_P = {
    0.6, 0.7, 0.8, 0.9
  }; // covers 1-p by symmetry; |r| = |2p-1|
  public static double[] ISO_CONFIDENCE_LINES_ALPHA = {1.0, 2.0, 3.0, 4.0}; // |alpha|

  public int which_isolines;

  // 0 : nothing, 1 : p, 2 : alpha

  public PoincareDiskEmbeddingUI(PoincareDiskEmbedding m) {
    boss = m;
    wait = true;
    tree_changed = false;
    which_isolines = 0;
  }

  public void paint(Graphics g, JComponent component) {
    if ((wait) && (boss.boss.myViewer.myExperiments.plot_ready)) {
      wait = false;
      boss.boss.myViewer.plotAvailable = true;

      boss.index_split_CV_plot = boss.boss.myViewer.myExperiments.index_split_CV_plot;
      boss.index_tree_number_plot = boss.boss.myViewer.myExperiments.index_tree_number_plot;
      boss.index_algorithm_plot = boss.boss.myViewer.myExperiments.index_algorithm_plot;
      tree_changed = true;

      System.out.println("On to plotting -- Keys:\n" + boss.codeKeys());
    }

    if (boss.boss.myViewer.plotAvailable) {
      PoincareDiskEmbedding pd = (PoincareDiskEmbedding) component;
      Graphics2D g2D = (Graphics2D) g;
      MonotonicTreeGraph mtg = null;
      double tree_p, tree_p_star, r;

      put_background(g2D, pd);

      if (boss.level_of_textual_details >= 2) {
        Dimension dimpd = pd.getSize();
        int n = Math.min(dimpd.height, dimpd.width);
        int radius = n / 2;
        double new_radius;
        int[] center = HyperbolicPoint2D_to_panel(new HyperbolicPoint2D(0.0, 0.0), dimpd);

        g2D.setPaint(PoincareDiskEmbeddingUI.LEVERAGING_COEFFICIENTS_COLOR);
        g2D.setStroke(new BasicStroke(2));

        tree_p =
            boss.boss.myViewer.myExperiments.myAlgos.all_algorithms.elementAt(
                    boss.index_algorithm_plot)
                .recordAllTrees[boss.index_split_CV_plot][boss.index_tree_number_plot]
                .tree_p_t;
        tree_p_star =
            boss.boss.myViewer.myExperiments.myAlgos.all_algorithms.elementAt(
                    boss.index_algorithm_plot)
                .recordAllTrees[boss.index_split_CV_plot][boss.index_tree_number_plot]
                .tree_p_t_star;

        r = Math.abs(2.0 * tree_p - 1.0);
        new_radius = ((double) radius) * r;
        g2D.drawOval(
            center[0] - (int) new_radius,
            center[1] - (int) new_radius,
            2 * (int) new_radius,
            2 * (int) new_radius);

        r = Math.abs(2.0 * tree_p_star - 1.0);
        new_radius = ((double) radius) * r;
        g2D.drawOval(
            center[0] - (int) new_radius,
            center[1] - (int) new_radius,
            2 * (int) new_radius,
            2 * (int) new_radius);
      }

      g2D.setColor(DISPLAY_FONT_COLOR);
      g2D.setFont(DISPLAY_FONT);

      if (DecisionTreeSkipTreeArc.ALL_ALPHA_TYPES[PoincareDiskEmbeddingUI.ALPHA_TYPE_PLOT].equals(
          DecisionTreeSkipTreeArc.USE_CARDINALS))
        mtg =
            boss.boss.myViewer.myExperiments.myAlgos.all_algorithms.elementAt(
                    boss.index_algorithm_plot)
                .recordAllMonotonicTreeGraphs_cardinals[boss.index_split_CV_plot][
                boss.index_tree_number_plot];
      else if (DecisionTreeSkipTreeArc.ALL_ALPHA_TYPES[PoincareDiskEmbeddingUI.ALPHA_TYPE_PLOT]
          .equals(DecisionTreeSkipTreeArc.USE_BOOSTING_WEIGHTS))
        mtg =
            boss.boss.myViewer.myExperiments.myAlgos.all_algorithms.elementAt(
                    boss.index_algorithm_plot)
                .recordAllMonotonicTreeGraphs_boosting_weights[boss.index_split_CV_plot][
                boss.index_tree_number_plot];
      else
        Dataset.perror(
            "MonotonicTreeGraph.class :: no such prediction as "
                + DecisionTreeSkipTreeArc.ALL_ALPHA_TYPES[PoincareDiskEmbeddingUI.ALPHA_TYPE_PLOT]);

      if (boss.level_of_textual_details >= 1)
        g2D.drawString(displayStringComponent(mtg), 0, DISPLAY_FONT_SIZE);

      plot_tree(g2D, pd, mtg);

      g.dispose();
    }
  }

  public int[] HyperbolicPoint2D_to_panel(HyperbolicPoint2D p, Dimension dim) {
    int n = Math.min(dim.height, dim.width);
    int w = dim.width, h = dim.height;

    int[] ret = new int[2];
    ret[0] = (int) (((p.x / 2.0) * (double) n) + ((double) w / 2.0));
    ret[1] = (int) (((-p.y / 2.0) * (double) n) + ((double) h / 2.0));

    return ret;
  }

  public String displayString(MonotonicTreeGraph mtg) {
    return "["
        + boss.index_split_CV_plot
        + ", "
        + boss.index_tree_number_plot
        + ", "
        + boss.index_algorithm_plot
        + "] ["
        + DecisionTreeSkipTreeArc.ALL_ALPHA_TYPES[PoincareDiskEmbeddingUI.ALPHA_TYPE_PLOT]
        + "] ["
        + DF0.format(mtg.expected_embedding_quality_error)
        + " %]";
  }

  public String displayString() {
    return "["
        + boss.index_split_CV_plot
        + ", "
        + boss.index_tree_number_plot
        + ", "
        + boss.index_algorithm_plot
        + "] ["
        + DecisionTreeSkipTreeArc.ALL_ALPHA_TYPES[PoincareDiskEmbeddingUI.ALPHA_TYPE_PLOT]
        + "] ["
        + boss.level_of_textual_details
        + "]";
  }

  public String displayStringComponent(MonotonicTreeGraph mtg) {
    return "[\u03C1: " + DF0.format(mtg.expected_embedding_quality_error) + " %]";
  }

  public void plot_tree(Graphics2D g2D, PoincareDiskEmbedding pd, MonotonicTreeGraph mtg) {
    Dimension dimpd = pd.getSize();
    int[] hp_panel;
    List<MonotonicTreeNode> pile = new ArrayList<>();
    pile.add(mtg.root);
    int i;
    boolean first = true;

    if (tree_changed) {
      System.out.println("Now plotting " + displayString(mtg) + " => " + mtg);
      tree_changed = false;
    }

    MonotonicTreeNode dumn;

    g2D.setFont(LABEL_FONT);
    while (pile.size() > 0) {
      dumn = pile.remove(0);
      hp_panel = HyperbolicPoint2D_to_panel(dumn.embedding_coordinates, dimpd);

      if (!dumn.is_leaf) {
        g2D.setPaint(COLOR_EDGE);
        for (i = 0; i < dumn.children_arcs.size(); i++) {
          g2D.setStroke(
              new BasicStroke(dumn.children_arcs.elementAt(i).path_from_start_to_end.length));
          // thickness proportional to the length of the DT's path mapped onto this arc

          drawHyperbolicEdge(
              g2D,
              pd,
              dumn.embedding_coordinates,
              dumn.children_arcs.elementAt(i).monotonic_end.embedding_coordinates);
          pile.add(dumn.children_arcs.elementAt(i).monotonic_end);
        }
      }

      if (SHOW_LABELS_WITH_COLORS) {
        if (dumn.alpha_value > 0) g2D.setPaint(LABEL_COLORS[0]);
        else g2D.setPaint(LABEL_COLORS[1]);
      } else g2D.setPaint(COLOR_VERTEX);

      if (first)
        g2D.fill3DRect(
            hp_panel[0] - PoincareDiskEmbeddingUI.VERTEX_RADIUS,
            hp_panel[1] - PoincareDiskEmbeddingUI.VERTEX_RADIUS,
            2 * PoincareDiskEmbeddingUI.VERTEX_RADIUS,
            2 * PoincareDiskEmbeddingUI.VERTEX_RADIUS,
            false);
      else
        g2D.fillOval(
            hp_panel[0] - PoincareDiskEmbeddingUI.VERTEX_RADIUS,
            hp_panel[1] - PoincareDiskEmbeddingUI.VERTEX_RADIUS,
            2 * PoincareDiskEmbeddingUI.VERTEX_RADIUS,
            2 * PoincareDiskEmbeddingUI.VERTEX_RADIUS);

      first = false;

      if (boss.level_of_textual_details == 1) { // REPLACE == 1 by >= 1 (done for visibility)
        g2D.setColor(LABEL_FONT_COLOR);
        g2D.drawString("#" + dumn.handle.name, hp_panel[0], hp_panel[1]);
      }
    }
  }

  public void put_background(Graphics2D g2D, PoincareDiskEmbedding pd) {
    Dimension dimpd = pd.getSize();

    int n = Math.min(dimpd.height, dimpd.width);
    int radius = n / 2;

    float color[] = PoincareDiskEmbeddingUI.COLORS[0].getColorComponents(null);
    Color bgc = new Color(color[0], color[1], color[2], 0.3f);

    g2D.setPaint(bgc);

    int[] center = HyperbolicPoint2D_to_panel(new HyperbolicPoint2D(0.0, 0.0), dimpd);

    g2D.setStroke(new BasicStroke(4));
    g2D.drawOval(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius);

    g2D.setPaint(bgc);
    int ratio = 10;
    int[] left = {center[0] - (radius / ratio), center[1]};
    int[] down = {center[0], center[1] + (radius / ratio)};
    int[] right = {center[0] + (radius / ratio), center[1]};
    int[] up = {center[0], center[1] - (radius / ratio)};

    g2D.drawLine(left[0], left[1], right[0], right[1]);
    g2D.drawLine(up[0], up[1], down[0], down[1]);

    if (which_isolines != 0) put_isolines(g2D, pd, dimpd, center, radius);
  }

  public void put_isolines(
      Graphics2D g2D, PoincareDiskEmbedding pd, Dimension dimpd, int[] center, int radius) {
    g2D.setPaint(PoincareDiskEmbeddingUI.ISO_CONFIDENCE_LINES_COLOR);
    g2D.setStroke(new BasicStroke(1));
    int i;
    double new_radius;

    if (which_isolines == 1)
      for (i = 0; i < ISO_CONFIDENCE_LINES_P.length; i++) {
        double r = Math.abs(2.0 * ISO_CONFIDENCE_LINES_P[i] - 1.0);

        new_radius = ((double) radius) * r;
        g2D.drawOval(
            center[0] - (int) new_radius,
            center[1] - (int) new_radius,
            2 * (int) new_radius,
            2 * (int) new_radius);
        g2D.drawString(
            ISO_CONFIDENCE_LINES_P_TEXT[i],
            (dimpd.width / 2) + (int) (new_radius * Math.cos(Math.PI / 4.0)),
            (dimpd.height / 2) - (int) (new_radius * Math.sin(Math.PI / 4.0)));
      }
    else if (which_isolines == 2)
      for (i = 0; i < ISO_CONFIDENCE_LINES_ALPHA.length; i++) {
        double r =
            (Math.exp(ISO_CONFIDENCE_LINES_ALPHA[i]) - 1.0)
                / (Math.exp(ISO_CONFIDENCE_LINES_ALPHA[i]) + 1.0);

        new_radius = ((double) radius) * r;
        g2D.drawOval(
            center[0] - (int) new_radius,
            center[1] - (int) new_radius,
            2 * (int) new_radius,
            2 * (int) new_radius);
        g2D.drawString(
            ISO_CONFIDENCE_LINES_ALPHA_TEXT[i],
            (dimpd.width / 2) + (int) (new_radius * Math.cos(Math.PI / 4.0)),
            (dimpd.height / 2) - (int) (new_radius * Math.sin(Math.PI / 4.0)));
      }
  }

  public boolean too_far(HyperbolicPoint2D p, HyperbolicPoint2D q, HyperbolicPoint2D center) {
    if (center.euclid_distance(p) > TOO_BIG_RATIO * q.euclid_distance(p)) return true;
    return false;
  }

  public void drawHyperbolicEdge(
      Graphics2D g2D, PoincareDiskEmbedding pd, HyperbolicPoint2D p, HyperbolicPoint2D q) {
    Dimension dimpd = pd.getSize();
    HyperbolicPoint2D r;
    if (p.euclid_norm() < q.euclid_norm()) r = PoincareDiskEmbeddingUI.CIRCLE_INVERSION(q);
    else r = PoincareDiskEmbeddingUI.CIRCLE_INVERSION(p);

    double den = p.x * (q.y - r.y) - p.y * (q.x - r.x) + (q.x * r.y - q.y * r.x),
        start_angle,
        delta_angle;

    int[] p_panel = HyperbolicPoint2D_to_panel(p, dimpd);
    int[] q_panel = HyperbolicPoint2D_to_panel(q, dimpd);

    g2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

    if (Statistics.APPROXIMATELY_EQUAL(den, 0.0, EPS3)) {
      // line
      g2D.drawLine(p_panel[0], p_panel[1], q_panel[0], q_panel[1]);
    } else {
      // arc
      double numx =
          p.euclid_norm_squared() * (q.y - r.y)
              - p.y * (q.euclid_norm_squared() - r.euclid_norm_squared())
              + (q.euclid_norm_squared() * r.y - r.euclid_norm_squared() * q.y);
      double numy =
          p.euclid_norm_squared() * (q.x - r.x)
              - p.x * (q.euclid_norm_squared() - r.euclid_norm_squared())
              + (q.euclid_norm_squared() * r.x - r.euclid_norm_squared() * q.x);

      HyperbolicPoint2D center = new HyperbolicPoint2D(numx / (2.0 * den), -numy / (2.0 * den));

      if (too_far(p, q, center)) g2D.drawLine(p_panel[0], p_panel[1], q_panel[0], q_panel[1]);
      else {

        int[] center_panel = HyperbolicPoint2D_to_panel(center, dimpd);

        double p_theta = Math.toDegrees(p.angle(center));
        double q_theta = Math.toDegrees(q.angle(center));

        double radius_panel =
            Math.sqrt(
                (double)
                    ((center_panel[0] - p_panel[0]) * (center_panel[0] - p_panel[0])
                        + (center_panel[1] - p_panel[1]) * (center_panel[1] - p_panel[1])));

        start_angle = p_theta;
        if (Math.abs(q_theta - p_theta) > 180.0)
          delta_angle = (360.0 - Math.abs(p_theta - q_theta)) * ((p_theta > q_theta) ? 1.0 : -1.0);
        else delta_angle = q_theta - p_theta;

        Arc2D arc =
            new Arc2D.Double(
                (double) center_panel[0] - radius_panel,
                (double) center_panel[1] - radius_panel,
                2.0 * radius_panel,
                2.0 * radius_panel,
                start_angle,
                delta_angle,
                Arc2D.OPEN);
        g2D.draw(arc);
      }
    }
  }

  public static HyperbolicPoint2D POINCARE_CIRCLE_INVERSION_ORIGIN_MODELE(
      HyperbolicPoint2D z, HyperbolicPoint2D x) {
    double normz2 = z.euclid_norm_squared();
    double zscal = 1.0 / normz2;
    double normx2 = x.euclid_norm_squared();
    double radius2 = zscal - 1.0;
    double xa2 = normx2 + zscal - 2 * zscal * z.dot(x);
    double scal = radius2 / xa2;

    HyperbolicPoint2D zscalz = z.times(zscal);
    HyperbolicPoint2D delta = x.subtract(zscalz);
    HyperbolicPoint2D left = delta.times(scal);

    HyperbolicPoint2D res = left.add(zscalz);

    return res;
  }

  public static HyperbolicPoint2D CIRCLE_INVERSION(HyperbolicPoint2D x) {
    // simple inversion wrt center of unit circle, at coordinates (0,0)

    double ix2 = 1.0 / x.euclid_norm_squared();
    return x.times(ix2);
  }

  public static HyperbolicPoint2D POINCARE_CIRCLE_INVERSION(
      HyperbolicPoint2D center, HyperbolicPoint2D x) {
    // Circle inversion wrt sphere (center, 1.0)

    return x.subtract(center).times(HyperbolicPoint2D.FACTOR_INVERSION(center, x)).add(center);
  }

  public static HyperbolicPoint2D POINCARE_CIRCLE_INVERSION_ORIGIN(
      HyperbolicPoint2D z, HyperbolicPoint2D x) {
    // Circle inversion wrt sphere (normalized center = (1/||z||^2).z, 1.0)

    return POINCARE_CIRCLE_INVERSION(HyperbolicPoint2D.POINCARE_NORMALIZATION(z), x);
  }
}
