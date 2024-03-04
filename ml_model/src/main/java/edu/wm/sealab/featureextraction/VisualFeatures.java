package edu.wm.sealab.featureextraction;

import java.util.ArrayList;

import lombok.Data;

public @Data class VisualFeatures {

  private ArrayList<ArrayList<Integer>> visualFeaturesMatrix = new ArrayList<ArrayList<Integer>>();

  private ArrayList<Double> featuresX = new ArrayList<Double>();
  private ArrayList<Double> featuresY = new ArrayList<Double>();

  public ArrayList<ArrayList<Integer>> getVisualFeaturesMatrix() {
    return visualFeaturesMatrix;
  }

  public void makeVisualFeaturesMatrix(String[] snippet) {
    for (int i = 0; i < snippet.length; i++) {
      visualFeaturesMatrix.add(new ArrayList<Integer>());
      int lineLength = snippet[i].length();
      for (int j = 0; j < lineLength-1; j++) {
        visualFeaturesMatrix.get(i).add(0);
      }
    }
  }

  public void findVisualXY() {
    for (int i = 1; i <= 7; i++) {
      featuresX.add(findVisualX(i));
      featuresY.add(findVisualY(i));
    }
    //Sum all literals for visual feature calculation
    featuresX.set(5, getNumbersX() + getStringsX() + getLiteralsX());
    featuresY.set(5, getNumbersY() + getStringsY() + getLiteralsY());
  }

  private double findVisualX(int featureNumber) {
    int sumX = 0;
    int sumFeatures = 0;
    double sumRatios = 0;
    for (ArrayList<Integer> line : visualFeaturesMatrix) {
      for (Integer num : line) {
        if (num != 0) {
          if (num == featureNumber) {
            sumX++;
          }
          sumFeatures++;
        }
      }
      sumRatios += (1.0 * sumX) / sumFeatures;
    }
    return sumRatios;
  }

  private double findVisualY(int featureNumber) {
    int sumY = 0;
    int sumFeatures = 0;
    double sumRatios = 0;
    int maxLineLength = findMaxLineLength();
    for (int c = 0; c < maxLineLength; c++) {
      for (ArrayList<Integer> line : visualFeaturesMatrix) {
        if (c < line.size()) {
          int num = line.get(c);
          if (num != 0) {
            if (num == featureNumber){
              sumY++;
            }
            sumFeatures++;
          }
        }
      }
      sumRatios += (1.0 * sumY) / sumFeatures;
    }
    return sumRatios;
  }

  private int findMaxLineLength() {
    int maxSize = 0;
    for (ArrayList<Integer> line : visualFeaturesMatrix) {
      if (line.size() > maxSize)
        maxSize = line.size();
    }
    return maxSize;
  }

  //Visual X getters
  public double getKeywordsX() {
    return featuresX.get(0);
  }

  public double getIdentifiersX() {
    return featuresX.get(1);
  }

  public double getOperatorsX() {
    return featuresX.get(2);
  }

  public double getNumbersX() {
    return featuresX.get(3);
  }

  public double getStringsX() {
    return featuresX.get(4);
  }

  public double getLiteralsX() {
    return featuresX.get(5);
  }

  public double getCommentsX() {
    return featuresX.get(6);
  }

  //Visual Y getters
  public double getKeywordsY() {
    return featuresY.get(0);
  }

  public double getIdentifiersY() {
    return featuresY.get(1);
  }

  public double getOperatorsY() {
    return featuresY.get(2);
  }

  public double getNumbersY() {
    return featuresY.get(3);
  }

  public double getStringsY() {
    return featuresY.get(4);
  }

  public double getLiteralsY() {
    return featuresY.get(5);
  }

  public double getCommentsY() {
    return featuresY.get(6);
  }

  public String printMatrix() {
    StringBuilder output = new StringBuilder();
    for (ArrayList<Integer> line : visualFeaturesMatrix) {
      output.append(line.toString());
      output.append("\n");
    }
    return output.toString();
  }

  public String visualXYString() {
    StringBuilder output = new StringBuilder();
    output.append(featuresX.toString());
    output.append("\n");
    output.append(featuresY.toString());
    return output.toString();
  }
}
