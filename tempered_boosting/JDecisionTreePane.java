import java.awt.event.*;
import java.awt.Color;
import java.awt.KeyboardFocusManager;
import java.awt.Component;
import java.awt.Toolkit;
import java.io.*;
import javax.swing.*;

class JDecisionTreePane extends JTabbedPane implements ActionListener, Debuggable {

  static int Max_Shadow_Color = 220;

  JDecisionTreeViewer myViewer;

  PoincareDiskEmbedding poincareDisk;

  JDecisionTreePane(JDecisionTreeViewer a) {
    super();
    myViewer = a;

    poincareDisk = new PoincareDiskEmbedding(this);
    poincareDisk.setBackground(Color.white);
    poincareDisk.addMouseWheelListener(poincareDisk);
    poincareDisk.addKeyListener(poincareDisk);
    poincareDisk.addComponentListener(
        new ComponentAdapter() {
          public void componentShown(ComponentEvent evt) {
            lookUpComponentShownPoincare(evt);
          }
        });

    addTab("Poincare Disk Embedding", null, poincareDisk);
    setBackgroundAt(0, Color.black);
  }

  private void lookUpComponentShownPoincare(ComponentEvent evt) {
    ((PoincareDiskEmbedding) getSelectedComponent()).requestFocus();
  }

  public void actionPerformed(ActionEvent e) {
    String ret = null;

    if (getSelectedIndex() < 2) {
      String command = e.getActionCommand();

      requestFocus();
      poincareDisk.requestFocus();
    }
  }
}
