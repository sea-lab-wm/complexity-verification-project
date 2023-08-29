package edu.wm.sealab.featureextraction;

import lombok.Data;

public @Data class Features {

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
}
