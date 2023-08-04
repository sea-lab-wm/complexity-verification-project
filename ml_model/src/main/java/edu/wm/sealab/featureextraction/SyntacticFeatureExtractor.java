package edu.wm.sealab.featureextraction;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SyntacticFeatureExtractor {

  private Features features = new Features();

  public SyntacticFeatureExtractor(Features features) {
    this.features = features;
  }

  /**
   * Extracts all the syntactic features into the given Feature Map
   *
   * @param snippet
   * @return the filled in Feature Map
   */


  public Features extract(String snippet) {

    features.setCommas(count(snippet, ","));
    features.setPeriods(count(snippet, "\\.") - 1);
    features.setSpaces(count(snippet, " "));
    features.setParenthesis(count(snippet, "\\(") * 2);

    //get the number of lines of code
    String[] lines = snippet.split("\n");
    int loc = lines.length;

    //average number of commas in each line
    int commas = count(snippet, ",");
    float avgCommas = commas / loc;
    features.setAvgCommas(avgCommas);
    //average number of parenthesis in each line
    int parenthesis = count(snippet, "\\(") * 2;
    float avgParenthesis = parenthesis / loc;
    features.setAvgParenthesis(avgParenthesis);
    //average number of periods in each line
    int periods = count(snippet, "\\.") - 1;
    float avgPeriods = periods / loc;
    features.setAvgPeriods(avgPeriods);
    //average number of spaces in each line
    int spaces = count(snippet, " ");
    float avgSpaces = spaces / loc;
    features.setAvgSpaces(avgSpaces);
    //average length in each line
    int totalLength = 0;
    for (String line : lines) {
        totalLength += line.length();
    }
    float avgLength = (float) totalLength / loc;
    features.setAvgLength(avgLength);
    
    // get the maximum length in any line
    int maxLength = 0;
    for (String line : lines) {
        int lineLength = line.length();
        if (lineLength > maxLength) {
            maxLength = lineLength;
        }
    }
    features.setMaxLength(maxLength);


    return features;
  }




  /**
   * Counts the occurences of the given character in the given string
   *
   * @param s
   * @param ch
   * @return the # of occurences
   */
  private int count(String s, String ch) {
    // Use Matcher class of java.util.regex
    // to match the character
    Matcher matcher = Pattern.compile(ch).matcher(s);
    int res = 0;

    while (matcher.find()) res++;

    return res;
  }
  

}
