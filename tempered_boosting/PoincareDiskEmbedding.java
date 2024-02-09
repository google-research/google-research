import java.awt.Dimension;
import java.awt.Image;
import java.awt.Color;
import java.awt.Rectangle;
import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.event.MouseMotionListener;
import java.awt.event.MouseWheelEvent;
import java.awt.event.MouseWheelListener;
import java.util.*;

import javax.swing.JPanel;

public class PoincareDiskEmbedding extends JPanel
    implements MouseMotionListener, KeyListener, MouseWheelListener, MouseListener {

  JDecisionTreePane boss;
  PoincareDiskEmbeddingUI assistant;

  boolean pressed;

  int index_split_CV_plot, index_tree_number_plot, index_algorithm_plot;

  public static int MAX_LEVEL_OF_TEXTUAL_DETAILS = 2;

  int level_of_textual_details = 2;

  // 0: no text, just isolines eventually
  // 1: 0+node labels + error
  // 2: 1+info on leveraging coefficients

  PoincareDiskEmbedding(JDecisionTreePane v) {
    boss = v;

    index_split_CV_plot = index_tree_number_plot = index_algorithm_plot = -1;

    addMouseMotionListener(this);
    addMouseListener(this);
    addKeyListener(this);
    addMouseWheelListener(this);
    assistant = new PoincareDiskEmbeddingUI(this);

    setFocusable(true);
    requestFocus();
    setUI(assistant);
  }

  public Rectangle getCaptureRectangle() {
    Rectangle bounds = getBounds();
    bounds.setLocation(getLocationOnScreen());
    return bounds;
  }

  public Dimension getMinimumSize() {
    return getPreferredSize();
  }

  public Dimension getPreferredSize() {
    return new Dimension(600, 600);
  }

  public Dimension getMaximumSize() {
    return getPreferredSize();
  }

  public void mouseDragged(MouseEvent arg0) {}

  public void mouseMoved(MouseEvent arg0) {
    repaint();
  }

  public void mouseEntered(MouseEvent arg0) {}

  public void mouseExited(MouseEvent arg0) {}

  public void mousePressed(MouseEvent arg0) {}

  public void mouseReleased(MouseEvent arg0) {}

  public void keyPressed(KeyEvent arg0) {}

  public String codeKeys() {
    return "===========================================================================================\n"
               + "T: switch tree in the (algorithm, split, tree) triple\n"
               + "S: switch split in the (algorithm, split, tree) triple\n"
               + "A: switch algorithm in the (algorithm, split, tree) triple\n"
               + "O: switch between using boosting weights and initial weights (cardinals) for"
               + " predictions\n"
               + "C: capture and save the currently displayed pane\n"
               + "I: change the display of isolines in (confidence wrt posterior, confidence wrt"
               + " alpha, none)\n"
               + "D: level of details in plot (see code for more)\n"
               + "===========================================================================================\n";
  }

  public void keyReleased(KeyEvent arg0) {
    if (!pressed) return;

    if (arg0.getKeyCode() == KeyEvent.VK_D) { // change level of textual details
      if (level_of_textual_details == PoincareDiskEmbedding.MAX_LEVEL_OF_TEXTUAL_DETAILS)
        level_of_textual_details = 0;
      else level_of_textual_details++;

      assistant.tree_changed = true;
    }

    if (arg0.getKeyCode() == KeyEvent.VK_T) { // keep #split, #algo, change #tree
      if (index_tree_number_plot
          == boss.myViewer.myExperiments.myAlgos.all_algorithms.elementAt(index_algorithm_plot)
                  .recordAllMonotonicTreeGraphs_cardinals[index_split_CV_plot]
                  .length
              - 1) index_tree_number_plot = 0;
      else index_tree_number_plot++;

      assistant.tree_changed = true;
    }

    if (arg0.getKeyCode() == KeyEvent.VK_S) { // keep #tree, #algo, change #split
      if (index_split_CV_plot
          == boss.myViewer.myExperiments.myAlgos.all_algorithms.elementAt(index_algorithm_plot)
                  .recordAllMonotonicTreeGraphs_cardinals
                  .length
              - 1) index_split_CV_plot = 0;
      else index_split_CV_plot++;

      assistant.tree_changed = true;
    }

    if (arg0.getKeyCode() == KeyEvent.VK_A) { // keep #tree, #split, change #algo
      if (boss.myViewer.myExperiments.myAlgos.all_algorithms.size() > 0) {
        int nextAlg = index_algorithm_plot + 1;
        if (nextAlg == boss.myViewer.myExperiments.myAlgos.all_algorithms.size()) nextAlg = 0;

        while ((nextAlg != index_algorithm_plot)
            && (!boss.myViewer
                .myExperiments
                .myAlgos
                .all_algorithms
                .elementAt(nextAlg)
                .name
                .equals((Boost.KEY_NAME_LOG_LOSS)))) nextAlg++;

        if (nextAlg != index_algorithm_plot) index_algorithm_plot = nextAlg;

        assistant.tree_changed = true;
      }
    }

    if (arg0.getKeyCode() == KeyEvent.VK_O) { // switch between ALPHA_TYPEs for plots
      PoincareDiskEmbeddingUI.ALPHA_TYPE_PLOT = 1 - PoincareDiskEmbeddingUI.ALPHA_TYPE_PLOT;

      assistant.tree_changed = true;
    }

    if (arg0.getKeyCode() == KeyEvent.VK_I) { // switch between isolines
      if (assistant.which_isolines == 2) assistant.which_isolines = 0;
      else assistant.which_isolines++;

      assistant.tree_changed = true;
    }

    if (arg0.getKeyCode() == KeyEvent.VK_C) {
      System.out.println("Capturing " + assistant.displayString() + " !");
      boss.myViewer.captureAndSave();
    }

    pressed = false;

    repaint();
  }

  public void keyTyped(KeyEvent e) {
    pressed = true;
  }

  public void mouseWheelMoved(MouseWheelEvent arg0) {}

  public void mouseClicked(MouseEvent arg0) {}
}
