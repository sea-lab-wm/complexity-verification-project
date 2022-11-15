/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 *    ModelPerformanceChart.java
 *    Copyright (C) 2004-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.beans;

import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.*;
import java.beans.PropertyVetoException;
import java.beans.VetoableChangeListener;
import java.beans.beancontext.BeanContext;
import java.beans.beancontext.BeanContextChild;
import java.io.Serializable;
/**
 * Bean that can be used for displaying threshold curves (e.g. ROC curves) and
 * scheme error plots
 * 
 * @author Mark Hall
 * @version $Revision$
 */
public class ModelPerformanceChart extends JPanel implements Serializable, BeanContextChild {

  /** for serialization */
  private static final long serialVersionUID = -4602034200071195924L;

  protected transient JFrame m_popupFrame;

  protected boolean m_framePoppedUp = false;

  private transient VisualizePanel m_visPanel;
  private Object m_masterPlot;
  private Object m_offscreenPlotData;

  //ADDED BY KOBI
  public void runAll() {
    performRequest("request");
  }

  /**
   * Global info for this bean
   * 
   * @return a <code>String</code> value
   */
  public String globalInfo() {
    return "Visualize performance charts (such as ROC).";
  }

  //  @Override // removed to allow compilation
  //SNIPPET_STARTS
  /**
   * Describe <code>performRequest</code> method here.
   * 
   * @param request a <code>String</code> value
   * @exception IllegalArgumentException if an error occurs
   */
  public void performRequest(String request) {
    if (request.compareTo("Show chart") == 0) {
      try {
        // popup visualize panel
        if (!m_framePoppedUp) {
          m_framePoppedUp = true;

          final javax.swing.JFrame jf = new javax.swing.JFrame(
            "Model Performance Chart");
          jf.setSize(800, 600);
          jf.getContentPane().setLayout(new BorderLayout());
          jf.getContentPane().add(m_visPanel, BorderLayout.CENTER);
          jf.addWindowListener(new java.awt.event.WindowAdapter() {
            @Override
            public void windowClosing(java.awt.event.WindowEvent e) {
              jf.dispose();
              m_framePoppedUp = false;
            }
          });
          jf.setVisible(true);
          m_popupFrame = jf;
        } else {
          m_popupFrame.toFront();
        }
      } catch (Exception ex) {
        ex.printStackTrace();
        m_framePoppedUp = false;
      }
    } else if (request.equals("Clear all plots")) {
      m_visPanel.removeAllPlots();
      m_visPanel.validate();
      m_visPanel.repaint();
      m_visPanel = null;
      m_masterPlot = null;
      m_offscreenPlotData = null;
    } else {
      throw new IllegalArgumentException(request
        + " not supported (Model Performance Chart)");
    }
  }
  //SNIPPET_END
  //SNIPPETS_END

  @Override
  public void setBeanContext(BeanContext bc) throws PropertyVetoException {

  }

  @Override
  public BeanContext getBeanContext() {
    return null;
  }

  @Override
  public void addVetoableChangeListener(String name, VetoableChangeListener vcl) {

  }

  @Override
  public void removeVetoableChangeListener(String name, VetoableChangeListener vcl) {

  }

  private class VisualizePanel extends Component {
    public void removeAllPlots() {

    }

    public void validate() {

    }

    public void repaint() {

    }
  }
}
