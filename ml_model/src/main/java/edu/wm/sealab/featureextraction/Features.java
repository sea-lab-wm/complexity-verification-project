package edu.wm.sealab.featureextraction;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;

import lombok.Data;

public @Data class Features {

  private final int KEYWORD_FEATURE_ID = 1;
  private final int IDENTIFIER_FEATURE_ID = 2;
  private final int PUNCTUATION_FEATURE_ID = 3;
  private final int EQUALS_FEATURE_ID = 4;
  private final int NUMBER_FEATURE_ID = 5;
  private final int STRING_FEATURE_ID = 6;
  private final int DELIMETER_FEATURE_ID = 7;
  private final int OPERATOR_FEATURE_ID = 8;

  private Map<String, List<Integer>> lineNumberMap = new HashMap<String, List<Integer>>();
  private SortedMap<String, List<Integer>> commaMap = new TreeMap<String, List<Integer>>();
  private SortedMap<String, List<Integer>> periodMap = new TreeMap<String, List<Integer>>();
  private SortedMap<String, List<Integer>> spacesMap = new TreeMap<String, List<Integer>>();
  private SortedMap<String, List<Integer>> parenthesisMap = new TreeMap<String, List<Integer>>();
  private SortedMap<String, List<Integer>> endParenthesisMap = new TreeMap<String, List<Integer>>();
  private SortedMap<String, List<Integer>> semicolonMap = new TreeMap<String, List<Integer>>();
  private SortedMap<String, List<Integer>> bracketMap = new TreeMap<String, List<Integer>>();
  private SortedMap<String, List<Integer>> equalsMap = new TreeMap<String, List<Integer>>();

  private ArrayList<ArrayList<Integer>> visualFeaturesMatrix = new ArrayList<ArrayList<Integer>>();

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

  // feature 14: avg #commas
  private float avgCommas = 0.0f;
  
  // feature 15: avg #parenthesis
  private float avgParenthesis = 0.0f;
  
  // feature 16: avg #perriods
  private float avgPeriods = 0.0f;
  
  // feature 17: avg #spaces
  private float avgSpaces = 0.0f;
  
  // feature 18: avg line length (beginning spaces + characters)
  private float avgLineLength = 0.0f;

  private int totalLineLength = 0;
  
  // feature 19: max line length
  private int maxLineLength = 0;
  
  private int totalIndentation = 0;
  
  // feature 20: avg indentation
  private float avgIndentation = 0;
  
  // feature 21: max indentation
  private int maxIndentation = 0;

  private int TotalBlankLines = 0;
  
  // feature 22: avg blank lines
  private float avgBlankLines = 0;
  
  // feature 23: #numbers
  private int numbers;

  // feature 24 #statements
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

  public void makeVisualFeaturesMatrix() {
    addMapToMatrix(commaMap, PUNCTUATION_FEATURE_ID);
    addMapToMatrix(periodMap, PUNCTUATION_FEATURE_ID);
    addMapToMatrix(parenthesisMap, PUNCTUATION_FEATURE_ID);
    addMapToMatrix(endParenthesisMap, PUNCTUATION_FEATURE_ID);
    addMapToMatrix(semicolonMap, PUNCTUATION_FEATURE_ID);
    addMapToMatrix(bracketMap, PUNCTUATION_FEATURE_ID);
    addMapToMatrix(equalsMap, EQUALS_FEATURE_ID);
  }

  private void addMapToMatrix(SortedMap<String, List<Integer>> symbolMap, int featureID) {
    Set<String> keySet = symbolMap.keySet();
    for (String lineNumber : keySet) {
      List<Integer> symbolIndexes = symbolMap.get(lineNumber);

      ArrayList<Integer> matrixLine;
      while (visualFeaturesMatrix.size() <= Integer.parseInt(lineNumber)){
        visualFeaturesMatrix.add(new ArrayList<Integer>());
      }
      matrixLine = visualFeaturesMatrix.get(Integer.parseInt(lineNumber));
      for (Integer i : symbolIndexes) {
        while (matrixLine.size() <= i) {
          matrixLine.add(0);
        }
        matrixLine.set(i, featureID); 
      }
      visualFeaturesMatrix.set(Integer.parseInt(lineNumber), matrixLine);
    }
  }
}
