package edu.wm.sealab.featureextraction;

import java.util.ArrayList;

import lombok.Data;

public @Data class VisualFeatures {

  private ArrayList<ArrayList<Integer>> visualFeaturesMatrix = new ArrayList<ArrayList<Integer>>();

  public ArrayList<ArrayList<Integer>> getVisualFeaturesMatrix() {
    return visualFeaturesMatrix;
  }

  public void makeVisualFeaturesMatrix(String[] snippet) {
    for (int i = 0; i < snippet.length; i++) {
      visualFeaturesMatrix.add(new ArrayList<Integer>());
      int lineLength = snippet[i].length();
      for (int j = 0; j < lineLength; j++) {
        visualFeaturesMatrix.get(i).add(0);
      }
    }
  }

  public String printMatrix() {
    StringBuilder output = new StringBuilder();
    for (ArrayList<Integer> line : visualFeaturesMatrix) {
      output.append(line.toString());
      output.append("\n");
    }
    return output.toString();
  }
}
