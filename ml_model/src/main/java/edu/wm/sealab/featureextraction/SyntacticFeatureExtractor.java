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
    //test
    int commas = count(snippet, ",");
    int periods = count(snippet, "\\.");
    int spaces = count(snippet, " ");
    int parenthesis = count(snippet, "\\(") * 2;

    features.setCommas(commas);
    features.setPeriods(periods);
    features.setSpaces(spaces);
    features.setParenthesis(parenthesis);

    //get the number of lines of code
    String[] lines = snippet.split(",");
    int loc = lines.length;

    //average number of commas in each line
    float avgCommas = (float) commas / loc;
    features.setAvgCommas(avgCommas);
    //average number of parenthesis in each line
    float avgParenthesis = loc;
    features.setAvgParenthesis(avgParenthesis);
    //average number of periods in each line
    float avgPeriods = (float) periods / loc;
    features.setAvgPeriods(avgPeriods);
    //average number of spaces in each line
    float avgSpaces = (float) spaces / loc;
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

    //indentation length
    int total_indentation = 0;
    int maxIndentation = 0;
    for (String line : lines) {
      int line_indentation = 0;
      for (int i = 0; i < line.length(); i++) {
          char c = line.charAt(i);
          if (c == ' ' || c == '\t') {
              line_indentation++;
          } else {
              break;
          }
      }
      total_indentation += line_indentation;
      
      // Update maximum indentation
      if (line_indentation > maxIndentation) {
          maxIndentation = line_indentation;
      }
    }
  
    // Average and maximum indentation
    float avgIndentation = (float) total_indentation / loc;
    features.setAvgIndentation(avgIndentation);
    features.setMaxIndentation(maxIndentation);
  
    //character max length
    
    //blank line average
    int blankLines = 0;
    for (String line : lines) {
        if (line.trim().isEmpty()) {
            blankLines++;
        }
    }
    float avgBlankLines = (float) blankLines / loc;
    features.setAvgBlankLines(avgBlankLines);

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
