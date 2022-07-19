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
 *    ClassifierPerformanceEvaluatorCustomizer.java
 *    Copyright (C) 2011-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.beans;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.util.ArrayList;
import java.util.List;

import javax.swing.JButton;
import javax.swing.JPanel;

/**
 * GUI customizer for the classifier performance evaluator component
 * 
 * @author Mark Hall (mhall{[at]}pentaho{[dot]}com)
 * @version $Revision$
 */
public class ClassifierPerformanceEvaluatorCustomizer extends JPanel {

  private ModifyListener m_modifyListener;
  private List<String> m_evaluationMetrics;
  private ModifyListener m_cpe;
  private ModifyListener m_parent;

  //SNIPPET_STARTS
  private void addButtons() {
    JButton okBut = new JButton("OK");
    JButton cancelBut = new JButton("Cancel");

    JPanel butHolder = new JPanel();
    butHolder.setLayout(new GridLayout(1, 2));
    butHolder.add(okBut);
    butHolder.add(cancelBut);
    add(butHolder, BorderLayout.SOUTH);

    okBut.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        if (m_modifyListener != null) {
          m_modifyListener.setModifiedStatus(
              ClassifierPerformanceEvaluatorCustomizer.this, true);
        }

        if (m_evaluationMetrics.size() > 0) {
          StringBuilder b = new StringBuilder();
          for (String s : m_evaluationMetrics) {
            b.append(s).append(",");
          }
          String newList = b.substring(0, b.length() - 1);
          m_cpe.setEvaluationMetricsToOutput(newList);
        }
        if (m_parent != null) {
          m_parent.dispose();
        }
      }
    });

    cancelBut.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {

        customizerClosing();
        if (m_parent != null) {
          m_parent.dispose();
        }
      }
    });
  }
  //SNIPPETS_END

  private void customizerClosing() {

  }

  private class ModifyListener {
    public void setModifiedStatus(ClassifierPerformanceEvaluatorCustomizer classifierPerformanceEvaluatorCustomizer, boolean b) {

    }

    public void setEvaluationMetricsToOutput(String newList) {

    }

    public void dispose() {

    }
  }
}
