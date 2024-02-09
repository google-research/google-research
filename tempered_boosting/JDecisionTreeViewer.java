import java.awt.AWTException;
import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Rectangle;
import java.awt.Robot;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import javax.swing.*;
import javax.swing.event.*;

public class JDecisionTreeViewer extends JFrame implements Runnable {

  Experiments myExperiments;

  JDecisionTreePane myTabbedPane;
  boolean plotAvailable;

  JDecisionTreeViewer() {
    super();
    plotAvailable = false;
  }

  JDecisionTreeViewer(String s) {
    super(s);
    plotAvailable = false;
  }

  public void start() {
    Thread th = new Thread(this);
    th.start();
  }

  public void go(Experiments e) {
    myExperiments = e;

    start();
    setVisible(true);
  }

  public void captureAndSave() {
    String nameSave =
        myExperiments.myAlgos.myDomain.myDS.pathSave
            + "treeplot_"
            + Utils.NOW
            + "_"
            + DecisionTreeSkipTreeArc.ALL_ALPHA_TYPES[PoincareDiskEmbeddingUI.ALPHA_TYPE_PLOT]
            + "_Algo"
            + myTabbedPane.poincareDisk.index_algorithm_plot
            + "_SplitCV"
            + myTabbedPane.poincareDisk.index_split_CV_plot
            + "_Tree"
            + myTabbedPane.poincareDisk.index_tree_number_plot
            + ".png";

    Rectangle rect = myTabbedPane.poincareDisk.getCaptureRectangle();
    BufferedImage fc = null;
    try {
      fc = new Robot().createScreenCapture(rect);
    } catch (AWTException a) {
    }

    File output = new File(nameSave);

    try {
      ImageIO.write(fc, "png", output);
    } catch (IOException a) {
    }
  }

  public void run() {
    Container pane = getContentPane();
    pane.setLayout(new BorderLayout());

    int ss = 600;

    myTabbedPane = new JDecisionTreePane(this);

    JPanel upperPane = new JPanel();
    upperPane.setLayout(new BorderLayout());
    upperPane.setPreferredSize(new Dimension(ss, ss));
    upperPane.setMinimumSize(new Dimension(ss, ss));
    upperPane.setMaximumSize(new Dimension(ss, ss));
    upperPane.add(myTabbedPane, BorderLayout.CENTER);

    pane.add(upperPane, BorderLayout.CENTER);
    pack();
  }
}
