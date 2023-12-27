package edu.wm.sealab.featureextraction;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
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
    
    int commas = count(snippet, ",", features.getCommaMap());
    int periods = count(snippet, "\\.", features.getPeriodMap());
    int spaces = count(snippet, " ", features.getSpacesMap());
    int parenthesis = count(snippet, "\\(", features.getParenthesisMap()) * 2;
    count(snippet, "\\)", features.getEndParenthesisMap());
    count(snippet, ";", features.getSemicolonMap());
    count(snippet, "\\{", features.getBracketMap());
    count(snippet, "\\}", features.getBracketMap());
    count(snippet, "\\[", features.getBracketMap());
    count(snippet, "\\]", features.getBracketMap());
    count(snippet, "=", features.getEqualsMap());


    features.setCommas(commas);
    features.setPeriods(periods);
    features.setSpaces(spaces);
    features.setParenthesis(parenthesis);

    /* the line length counts all the characters of the given line 
    (including the spaces/tabs/indentation in the beginning) */
    String[] lines = snippet.split("\n");
    int totalLength = 0;
    for (String line : lines) {
        totalLength += line.length();
    }
    features.setTotalLineLength(totalLength);
    
    // get the maximum length in any line
    int maxLineLength = 0;
    for (String line : lines) {
      int lineLength = line.length();
      if (lineLength > maxLineLength) {
          maxLineLength = lineLength;
      }
    }
    features.setMaxLineLength(maxLineLength);

    //indentation length
    /* iterates through each line character by character, 
    counting spaces and tabs until it reaches a non-space, non-tab character */
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
      features.setTotalIndentation(total_indentation);
      // Update maximum indentation
      if (line_indentation > maxIndentation) {
          maxIndentation = line_indentation;
      }
    }
  
    // Maximum indentation
    features.setMaxIndentation(maxIndentation);
    
    // Blank line
    int blankLines = 0;
    for (String line : lines) {
        if (line.trim().isEmpty()) {
            blankLines++;
        }
    }

    features.setTotalBlankLines(blankLines);

    return features;
  }

  /**
   * Counts the occurences of the given character in the given string
   *
   * @param s
   * @param ch
   * @return the # of occurences
   */
  private int count(String s, String ch, Map<String, List<Integer>> symbolMap) {
    //Split snippet up into lines
    String[] lines = s.split("\n");
    int res = 0;
    //count features in each line
    for (int i = 0; i < lines.length; i++) {
      res += countLine(lines[i], ch, symbolMap, i+1);
    }
    return res;
  }

  private int countLine(String s, String ch, Map<String, List<Integer>> symbolMap, int lineIndex) {
    // Use Matcher class of java.util.regex
    // to match the character
    Matcher matcher = Pattern.compile(ch).matcher(s);
    int res = 0;
    //Add symbols to their symbol map for visual features
    String lineNumber = Integer.toString(lineIndex - 1);
    while (matcher.find()) { 
      res++;
      int symbolIndex = (matcher.end() - 1);

      List<Integer> list;
      if (symbolMap.containsKey(lineNumber)){
        list = symbolMap.get(lineNumber);
      }
      else {
        list = new ArrayList<>();
      }
      list.add(symbolIndex);
      symbolMap.put(lineNumber,list);
      
    }
    return res;
  }
}
