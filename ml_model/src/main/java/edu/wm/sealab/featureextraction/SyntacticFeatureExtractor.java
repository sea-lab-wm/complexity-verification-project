package edu.wm.sealab.featureextraction;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class SyntacticFeatureExtractor {

    private FeatureMap featureMap = new FeatureMap();

    public SyntacticFeatureExtractor(FeatureMap featureMap) {
        this.featureMap = featureMap;
    }

    /**
     * Extracts all the syntactic features into the given Feature Map
     * @param snippet
     * @return the filled in Feature Map
     */
    public FeatureMap extract(String snippet) {
        featureMap.setCommas(count(snippet, ","));
        featureMap.setPeriods(count(snippet, "\\.") - 1);
        featureMap.setSpaces(count(snippet, " "));
        featureMap.setParenthesis(count(snippet, "\\("));

        return featureMap;
    }

    /**
     * Counts the occurences of the given character in the given string
     * @param s
     * @param ch
     * @return the # of occurences
     */
    private int count(String s, String ch) {
        // Use Matcher class of java.util.regex
        // to match the character
        Matcher matcher = Pattern.compile(ch).matcher(s);
        int res = 0;

        while (matcher.find())
            res++;
  
        return res;
    }
}
