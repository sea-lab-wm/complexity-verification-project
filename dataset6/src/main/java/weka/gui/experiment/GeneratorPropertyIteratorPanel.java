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
 *    GeneratorPropertyIteratorPanel.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.gui.experiment;

import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.ArrayList;

import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JPanel;

/**
 * This panel controls setting a list of values for an arbitrary resultgenerator
 * property for an experiment to iterate over.
 * 
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class GeneratorPropertyIteratorPanel extends JPanel implements
  ActionListener {

  /** for serialization */
  private static final long serialVersionUID = -6026938995241632139L;

  /** Click to select the property to iterate over */
  protected JButton m_ConfigureBut = new JButton("Select property...");

  /** Controls whether the custom iterator is used or not */
  protected JComboBox m_StatusBox = new JComboBox();

  /** Allows editing of the custom property values */
  protected GenericArrayEditor m_ArrayEditor = new GenericArrayEditor();

  /** The experiment this all applies to */
  protected Experiment m_Exp;

  /**
   * Listeners who want to be notified about editing status of this panel
   */
  protected ArrayList<ActionListener> m_Listeners = new ArrayList<ActionListener>();


  //ADDED BY US
  public void runAll() {
    actionPerformed(new ActionEvent(new Object(), 1, "name"));
  }


  /**
   * Handles the various button clicking type activities.
   * 
   * @param e a value of type 'ActionEvent'
   */
  //SNIPPET_STARTS
  @Override
  public void actionPerformed(ActionEvent e) {

    if (e.getSource() == m_ConfigureBut) {
      selectProperty();
    } else if (e.getSource() == m_StatusBox) {
      // notify any listeners
      for (int i = 0; i < m_Listeners.size(); i++) {
        ActionListener temp = (m_Listeners.get(i));
        temp.actionPerformed(new ActionEvent(this,
          ActionEvent.ACTION_PERFORMED, "Editor status change"));
      }

      // Toggles whether the custom property is used
      if (m_StatusBox.getSelectedIndex() == 0) {
        m_Exp.setUsePropertyIterator(false);
        m_ConfigureBut.setEnabled(false);
        m_ArrayEditor.getCustomEditor().setEnabled(false);
        m_ArrayEditor.setValue(null);
        validate();
      } else {
        if (m_Exp.getPropertyArray() == null) {
          selectProperty();
        }
        if (m_Exp.getPropertyArray() == null) {
          m_StatusBox.setSelectedIndex(0);
        } else {
          m_Exp.setUsePropertyIterator(true);
          m_ConfigureBut.setEnabled(true);
          m_ArrayEditor.getCustomEditor().setEnabled(true);
        }
        validate();
      }
    }
  }
  //SNIPPETS_END

  private void selectProperty() {

  }

  private class Experiment {
    public void setUsePropertyIterator(boolean b) {

    }

    public Object getPropertyArray() {
      return null;
    }
  }

  private class GenericArrayEditor {
    public Component getCustomEditor() {
      return null;
    }

    public void setValue(Object o) {

    }
  }
}
