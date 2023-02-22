package FeatureExtraction;

public class FeatureMap {
   
    // feature 1: #parameters of method
    private  int numOfParameters = 0;

    // feature 2: #if statements
    private int numOfIfStatements = 0;

    // feature 3: #loops
    private int numOfLoops = 0;

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
}
