package edu.wm.sealab.featureextraction;

import lombok.Data;

public @Data class FeatureMap {
   
    // feature 1: #parameters of method
    private  int numOfParameters = 0;

    // feature 2: #if statements
    private int numOfIfStatements = 0;

    // feature 3: #loops
    private int numOfLoops = 0;

}
