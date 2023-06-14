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
 *    EstimatorUtils.java
 *    Copyright (C) 2004-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.estimators;

import java.util.Enumeration;

/**
 * Contains static utility functions for Estimators.
 * <p>
 * 
 * @author Gabi Schmidberger (gabi@cs.waikato.ac.nz)
 * @version $Revision$
 */
public class EstimatorUtils implements RevisionHandler {

  //ADDED BY KOBI
  public void runAll() {
    try {
      getMinMax(new Instances(), 1, new double[5]);
    } catch (Exception e) {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

  /**
   * Find the minimum and the maximum of the attribute and return it in the last
   * parameter..
   *
   * @param inst      instances used to build the estimator
   * @param attrIndex index of the attribute
   * @param minMax    the array to return minimum and maximum in
   * @return number of not missing values
   * @throws Exception if parameter minMax wasn't initialized properly
   */ // s47
  //SNIPPET_STARTS
  public static int getMinMax(Instances inst, int attrIndex, double[] minMax)
          throws Exception {
    double min = Double.NaN;
    double max = Double.NaN;
    Instance instance = null;
    int numNotMissing = 0;
    if ((minMax == null) || (minMax.length < 2)) {
      throw new Exception("Error in Program, privat method getMinMax");
    }

    Enumeration<Instance> enumInst = inst.enumerateInstances();
    if (enumInst.hasMoreElements()) {
      do {
        instance = enumInst.nextElement();
      } while (instance.isMissing(attrIndex) && (enumInst.hasMoreElements()));

      // add values if not missing
      if (!instance.isMissing(attrIndex)) {
        numNotMissing++;
        min = instance.value(attrIndex);
        max = instance.value(attrIndex);
      }
      while (enumInst.hasMoreElements()) {
        instance = enumInst.nextElement();
        if (!instance.isMissing(attrIndex)) {
          numNotMissing++;
          if (instance.value(attrIndex) < min) {
            min = (instance.value(attrIndex));
          } else {
            if (instance.value(attrIndex) > max) {
              max = (instance.value(attrIndex));
            }
          }
        }
      }
    }
    minMax[0] = min;
    minMax[1] = max;
    return numNotMissing;
  }
  //SNIPPETS_END

  private static class Instances {
    public Enumeration<Instance> enumerateInstances() {
      return null;
    }
  }

  private static class Instance {
    public boolean isMissing(int attrIndex) {
      return false;
    }

    public double value(int attrIndex) {
      return 0;
    }
  }
}