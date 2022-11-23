//using System.Collections.Generic;
import java.util.HashMap;
import java.util.*;
import java.io.*;
public class HistogramSemanticNormal
{
    // Histogram: Returns frequency counts of characters in a string for creating a histogram.
    // corpus: text corpus to analyse
    // frequencies: frequencies
    // index: index
    // character: character

    public static HashMap<Character, Integer> Histogram(String corpus)
    {
        HashMap<Character, Integer> frequencies = new HashMap<Character, Integer>();
        for (int index = 0; index < corpus.length(); index++)
        {
            char character = corpus.charAt(index);
            if (!frequencies.containsKey(character))
            {
                frequencies.put(character, 0);//dont know if this 0 is 0 or is 0 a polaceholder
            }
	    frequencies.put(character, frequencies.get(character)+1);            
        }
        return frequencies;
    }
}
