package edu.wm.sealab.featureextraction;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import lombok.Data;

public @Data class Features {

  private Map<String, List<Integer>> lineNumberMap = new HashMap<String, List<Integer>>();

  // feature 1: #parameters of method
  private int numOfParameters = 0;

  // feature 2: #if statements
  private int numOfIfStatements = 0;

  // feature 3: #loops
  private int numOfLoops = 0;

  // feature 4: #assignments expressions
  private int assignExprs = 0;

  // feature 5: #commas
  private int commas = 0;

  // feature 6: #periods
  private int periods = 0;

  // feature 7: #spaces
  private int spaces = 0;

  // feature 8: #comparisons
  private int comparisons = 0;

  // feature 9: #parenthesis
  private int parenthesis = 0;

  // feature 10: #literals
  private int literals = 0;

  // feature 11: #comments
  private int numOfComments = 0;

  // feature 12: #arithmeticOperators
  private int arithmeticOperators;

  // feature 13: #conditionals (if and switch statments)
  private int conditionals;

  // feature 14: #numbers
  private int numbers;

  // feature 14 #statements
  private int statements;

  public void incrementNumOfIfStatements() {
    setNumOfIfStatements(getNumOfIfStatements() + 1);
  }

  public void incrementNumOfLoops() {
    setNumOfLoops(getNumOfLoops() + 1);
  }

  public void incrementNumOfLiterals() {
    setLiterals(getLiterals() + 1);
  }

  public void incrementNumOfComments() {
    setNumOfComments(getNumOfComments() + 1);
  }

  public void incrementNumOfArithmeticOperators() {
    setArithmeticOperators(getArithmeticOperators() + 1);
  }

  public void incrementNumOfConditionals() {
    setConditionals(getConditionals() + 1);
  }

  public void incrementNumOfNumbers() {
    setNumbers(getNumbers() + 1);
  }

  public void incrementNumOfStatements() {
    setStatements(getStatements() + 1);
  }

  /**
   * Searches through features.lineNumberMap, a HashMap<String line_number, List<Integer> lineIntegers>
   * to find the line with the most integers, or the List<Integer> with the largest size
   */
  public int findMaxNumbers() {
    int max = 0;
    for (List<Integer> line : lineNumberMap.values()) {
      if (line.size() > max) {
        max = line.size();
      }
    }
    return max;
  }
}
