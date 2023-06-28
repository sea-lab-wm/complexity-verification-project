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
     * @param snippet
     * @return the filled in Feature Map
     */
    public Features extract(String snippet) {
        features.setCommas(count(snippet, ","));
        features.setPeriods(count(snippet, "\\.") - 1);
        features.setSpaces(count(snippet, " "));
        features.setParenthesis(count(snippet, "\\("));

        return features;
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
