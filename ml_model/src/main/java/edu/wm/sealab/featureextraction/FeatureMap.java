package edu.wm.sealab.featureextraction;

//import lombok.Data;

//public @Data class FeatureMap {
public class FeatureMap {
   
    // feature 1: #parameters of method
    private  int numOfParameters = 0;

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

    public int getNumOfParameters() {
        return numOfParameters;
    }

    public void setNumOfParameters(int numOfParameters) {
        this.numOfParameters = numOfParameters;
    }

    public int getNumOfIfStatements() {
        return numOfIfStatements;
    }

    public void setNumOfIfStatements(int numOfIfStatements) {
        this.numOfIfStatements = numOfIfStatements;
    }

    public int getNumOfLoops() {
        return numOfLoops;
    }

    public void setNumOfLoops(int numOfLoops) {
        this.numOfLoops = numOfLoops;
    }

    public int getNumOfAssignExprs() {
        return assignExprs;
    }

    public void setNumOfAssignExprs(int assignExprs) {
        this.assignExprs = assignExprs;
    }

    public int getNumOfCommas() {
        return commas;
    }

    public void setNumOfCommas(int commas) {
        this.commas = commas;
    }

    public int getNumOfPeriods() {
        return periods;
    }

    public void setNumOfPeriods(int periods) {
        this.periods = periods;
    }

    public int getNumOfSpaces() {
        return spaces;
    }

    public void setNumOfSpaces(int spaces) {
        this.spaces = spaces;
    }

    public int getNumOfComparisons() {
        return comparisons;
    }

    public void setNumOfComparisons(int comparisons) {
        this.comparisons = comparisons;
    }

    public int getNumOfParenthesis() {
        return parenthesis;
    }

    public void setNumOfParenthesis(int parenthesis) {
        this.parenthesis = parenthesis;
    }

    public int getNumOfLiterals() {
        return literals;
    }

    public void setNumOfLiterals(int literals) {
        this.literals = literals;
    }
}
