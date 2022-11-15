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
 *    CheckEstimator.java
 *    Copyright (C) 1999-2012 University of Waikato, Hamilton, New Zealand
 *
 */

package weka.estimators;

/**
 * Class for examining the capabilities and finding problems with estimators. If
 * you implement a estimator using the WEKA.libraries, you should run the checks
 * on it to ensure robustness and correct operation. Passing all the tests of
 * this object does not mean bugs in the estimator don't exist, but this will
 * help find some common ones.
 * <p/>
 * 
 * Typical usage:
 * <p/>
 * <code>java weka.estimators.CheckEstimator -W estimator_name 
 * estimator_options </code>
 * <p/>
 * 
 * This class uses code from the CheckEstimatorClass ATTENTION! Current
 * estimators can only 1. split on a nominal class attribute 2. build estimators
 * for nominal and numeric attributes 3. build estimators independendly of the
 * class type The functionality to test on other class and attribute types is
 * left in big parts in the code.
 * 
 * CheckEstimator reports on the following:
 * <ul>
 * <li>Estimator abilities
 * <ul>
 * <li>Possible command line options to the estimator</li>
 * <li>Whether the estimator can predict nominal, numeric, string, date or
 * relational class attributes. Warnings will be displayed if performance is
 * worse than ZeroR</li>
 * <li>Whether the estimator can be trained incrementally</li>
 * <li>Whether the estimator can build estimates for numeric attributes</li>
 * <li>Whether the estimator can handle nominal attributes</li>
 * <li>Whether the estimator can handle string attributes</li>
 * <li>Whether the estimator can handle date attributes</li>
 * <li>Whether the estimator can handle relational attributes</li>
 * <li>Whether the estimator build estimates for multi-instance data</li>
 * <li>Whether the estimator can handle missing attribute values</li>
 * <li>Whether the estimator can handle missing class values</li>
 * <li>Whether a nominal estimator only handles 2 class problems</li>
 * <li>Whether the estimator can handle instance weights</li>
 * </ul>
 * </li>
 * <li>Correct functioning
 * <ul>
 * <li>Correct initialisation during addvalues (i.e. no result changes when
 * addValues called repeatedly)</li>
 * <li>Whether incremental training produces the same results as during
 * non-incremental training (which may or may not be OK)</li>
 * <li>Whether the estimator alters the data pased to it (number of instances,
 * instance order, instance weights, etc)</li>
 * </ul>
 * </li>
 * <li>Degenerate cases
 * <ul>
 * <li>building estimator with zero training instances</li>
 * <li>all but one attribute attribute values missing</li>
 * <li>all attribute attribute values missing</li>
 * <li>all but one class values missing</li>
 * <li>all class values missing</li>
 * </ul>
 * </li>
 * </ul>
 * Running CheckEstimator with the debug option set will output the training and
 * test datasets for any failed tests.
 * <p/>
 * 
 * The <code>weka.estimators.AbstractEstimatorTest</code> uses this class to
 * test all the estimators. Any changes here, have to be checked in that
 * abstract test class, too.
 * <p/>
 * 
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 * -D
 *  Turn on debugging output.
 * </pre>
 * 
 * <pre>
 * -S
 *  Silent mode - prints nothing to stdout.
 * </pre>
 * 
 * <pre>
 * -N &lt;num&gt;
 *  The number of instances in the datasets (default 100).
 * </pre>
 * 
 * <pre>
 * -W
 *  Full name of the estimator analysed.
 *  eg: weka.estimators.NormalEstimator
 * </pre>
 * 
 * <pre>
 * Options specific to estimator weka.estimators.NormalEstimator:
 * </pre>
 * 
 * <pre>
 * -D
 *  If set, estimator is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <!-- options-end -->
 * 
 * Options after -- are passed to the designated estimator.
 * <p/>
 * 
 * @author Len Trigg (trigg@cs.waikato.ac.nz)
 * @author FracPete (fracpete at waikato dot ac dot nz)
 * @version $Revision$
 */
public class CheckEstimator implements OptionHandler, RevisionHandler {

  //ADDED BY KOBI
  public void runAll() {
    printAttributeSummary(1, 1);
  }

  /**
   * Print out a short summary string for the dataset characteristics
   * 
   * @param attrTypes the attribute types used (NUMERIC, NOMINAL, etc.)
   * @param classType the class type (NUMERIC, NOMINAL, etc.)
   */
  protected void printAttributeSummary(AttrTypes attrTypes, int classType) {

    String str = "";

    if (attrTypes.numeric) {
      str += " numeric";
    }

    if (attrTypes.nominal) {
      if (str.length() > 0) {
        str += " &";
      }
      str += " nominal";
    }

    if (attrTypes.string) {
      if (str.length() > 0) {
        str += " &";
      }
      str += " string";
    }

    if (attrTypes.date) {
      if (str.length() > 0) {
        str += " &";
      }
      str += " date";
    }

    if (attrTypes.relational) {
      if (str.length() > 0) {
        str += " &";
      }
      str += " relational";
    }

    str += " attributes)";

    switch (classType) {
    case Attribute.NUMERIC:
      str = " (numeric class," + str;
      break;
    case Attribute.NOMINAL:
      str = " (nominal class," + str;
      break;
    case Attribute.STRING:
      str = " (string class," + str;
      break;
    case Attribute.DATE:
      str = " (date class," + str;
      break;
    case Attribute.RELATIONAL:
      str = " (relational class," + str;
      break;
    }

    print(str);
  }

  private void print(String str) {
  }

  //SNIPPET_STARTS
  /**
   * Print out a short summary string for the dataset characteristics
   * 
   * @param attrType the attribute type (NUMERIC, NOMINAL, etc.)
   * @param classType the class type (NUMERIC, NOMINAL, etc.)
   */
  protected void printAttributeSummary(int attrType, int classType) {

    String str = "";

    switch (attrType) {
    case Attribute.NUMERIC:
      str = " numeric" + str;
      break;
    case Attribute.NOMINAL:
      str = " nominal" + str;
      break;
    case Attribute.STRING:
      str = " string" + str;
      break;
    case Attribute.DATE:
      str = " date" + str;
      break;
    case Attribute.RELATIONAL:
      str = " relational" + str;
      break;
    }
    str += " attribute(s))";

    switch (classType) {
    case Attribute.NUMERIC:
      str = " (numeric class," + str;
      break;
    case Attribute.NOMINAL:
      str = " (nominal class," + str;
      break;
    case Attribute.STRING:
      str = " (string class," + str;
      break;
    case Attribute.DATE:
      str = " (date class," + str;
      break;
    case Attribute.RELATIONAL:
      str = " (relational class," + str;
      break;
    }

    print(str);
  }
  //SNIPPET_END
  //SNIPPETS_END

  private class AttrTypes {
    public boolean numeric;
    public boolean nominal;
    public boolean string;
    public boolean date;
    public boolean relational;
  }

  private class Attribute {
    public static final int NUMERIC = 0;
    public static final int NOMINAL = 1;
    public static final int STRING = 2;
    public static final int DATE = 3;
    public static final int RELATIONAL = 4;
  }
}
