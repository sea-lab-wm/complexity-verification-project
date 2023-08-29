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

<<<<<<< HEAD
  // new features
  private float avgCommas = 0.0f;
  private float avgParenthesis = 0.0f;
  private float avgPeriods = 0.0f;
  private float avgSpaces = 0.0f;
  private float avgLength = 0.0f;

  private int maxLength = 0;

  private float avgIndentation = 0;
  private int maxIndentation = 0;

  private float avgBlankLines = 0;


=======
>>>>>>> 2624f3d5b8c60c0dda42a6e42af510440345845a
  public void incrementNumOfIfStatements() {
    setNumOfIfStatements(getNumOfIfStatements() + 1);
  }

  public void incrementNumOfLoops() {
    setNumOfLoops(getNumOfLoops() + 1);
  }

  public void incrementNumOfLiterals() {
    setLiterals(getLiterals() + 1);
  }
}
