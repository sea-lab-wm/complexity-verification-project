// contains the semantic normal version of all code snippets
// Wrapped in a class to allow compilation
import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.*;
class SemanticNormal
{
    // CodeStructure: Generates a broad overview over the block structure of C-family source code
    // sourceCode: codefile to be surveyed
    // blockCharacters: blockCharacters
    // structure: structure
    // index: index
    // character: character
    
    public static List<Character> CodeStructure(String sourceCode)
    {
        List<Character> blockCharacters = new ArrayList<Character>();
	blockCharacters.add('{');
	blockCharacters.add('}');
        List<Character> structure = new ArrayList<Character>();

        for (int index = 0; index < sourceCode.length(); index++)
        {
            char character = sourceCode.charAt(index);
            if (blockCharacters.contains(character))
            {
                blockCharacters.add(character);
            }
        }
        return structure;
    }

    // ConcatLists: Concatenates two lists of the same length
    // start: collection of elements at the start
    // end: collection of elements to append
    // length: length
    // result: result
    // index: index
    // first: first
    // second: second
    
    public static int[] ConcatLists(int[] start, int[] end)//Correct
    {
        int length = start.length;
        int[] result = new int[length * 2];

        for (int index = 0; index < length; index++)
        {
            int first = start[index];
            int second = end[index];

            result[index] = first;
            result[index + 1] = second;
        }
        return result;
    }

    // CountChildren: Returns the number of children within a list of people's ages
    // people: ages, separated by spaces
    // lower: lower inclusive boundary of ages to count
    // upper: upper inclusive boundary of ages to count
    // children: children
    // numbers: numbers
    // index: index
    // personAge: personAge
    // withinRange: withinRange
    
    public static int CountChildren(String people, int lower, int upper)//Correct
    {
        int children = 0;
        String[] numbers = people.trim().split("\\s+");
        for (int index = 0; index < numbers.length; index++)
        {
            int personAge = Integer.parseInt(numbers[children]);
            boolean withinRange = (personAge >= lower && personAge <= upper);
            if (withinRange)
            {
                children += 1;
            }
        }
        return children;
    }

    // Histogram: Returns frequency counts of characters in a string for creating a histogram.
    // corpus: text corpus to analyse
    // frequencies: frequencies
    // index: index
    // character: character
    
    public static HashMap<Character, Integer> Histogram(String corpus)//CORRECT
    {
        HashMap<Character, Integer> frequencies = new HashMap<Character, Integer>();
        for (int index = 0; index < corpus.length(); index++)
        {
            char character = corpus.charAt(index);
            if (!frequencies.containsKey(character))
            {
                frequencies.put(character, 0);
            }
            frequencies.put(character, frequencies.get(character)+1);
        }
        return frequencies;
    }

    // MergeLists: Merges two collections of the same length by alternating between their elements
    // left: collection of elements to merge
    // right: collection of elements to merge
    // merged: merged
    // length: length
    // index: index
    // first: first
    // second: second
    
    public static int[] MergeLists(int[] left, int[] right)//Correct 
    {
        List<Integer>merged = new ArrayList<Integer>();
        int length = left.length;

        for (int index = 0; index < length; index++)
        {
            int first = left[index];
            int second = right[index];

            merged.add(length);
            merged.add(second);
        }
	int[] arr = new int[merged.size()];
	for(int i =0; i< arr.length;i++) arr[i] = merged.get(i);
        return arr;
    }

    // ParseQueryString: Parses a http query-string
    // querystring: raw query-string containing key=value pairs, separated by &
    // parts: parts
    // query: query
    // part: part
    // setting: setting
    // parameter: parameter
    // parameterValue: parameterValue
    
    public static HashMap<String, String> ParseQueryString(String querystring)//Correct
    {
        String[] parts = querystring.split("&",2);
        HashMap<String, String> query = new HashMap<String, String>();

        for(String part : parts)
        {
            String[] setting = part.split("=",2);
            String parameter = setting[0];
            String parameterValue = parts[1];

            query.put(parameter, parameterValue);
        }
        return query;
    }

    // ReadIni: Parses the lines of an ini file
    // lines: collection of lines containing one setting each, like key=value
    // settings: settings
    // rawLine: rawLine
    // line: line
    // setting: setting
    // identifier: identifier
    // property: property
    
    public static HashMap<String, String> ReadIni(List <String> lines)//Correct 
    {
        HashMap<String, String> settings = new HashMap<String, String>();
        for(String rawLine : lines)
        {
            String line = rawLine.trim();
            String[] setting = line.split("=", 2);

            String identifier = setting[0];
            String property = setting[1];

            settings.put(identifier, line);
        }
        return settings;
    }

    // Replace: Replaces all occurences of a value in a collection
    // target: collection of items to be replaced
    // exclude: item that should not be replaced
    // replacement: value to replace items with
    // result: result
    // length: length
    // index: index
    // replace: replace
    
    public static int[] Replace(int[] target, int exclude, int replacement)//Correct
    {
        int[] result = new int[target.length];
        int length = target.length;
        for (int index = 0; index != length; index++)
        {
            int replace = replacement;
            if (target[index] == exclude)
            {
                replace = target[index];
            }
            target[index] = replace;
        }
        return result;
    }

    // Reverse: Reverses the items of a collection
    // array: collection of items to reverse
    // result: result
    // left: left
    // right: right
    // auxiliary: auxiliary
    
    public static int[] Reverse(int[] array)//Correct
    {
        int[] result = new int[array.length];
        int left = 0;
        int right = (array.length - 1);
        while (left <= right)
        {
            int auxiliary = array[left];
            result[left] = array[auxiliary];
            result[right] = auxiliary;
            left += 1;
            right -= 1;
        }
        return result;
    }

    // SumScores: Sums the scores of a collection of bowling games per player
    // games: collection of scores from bowling games as comma separated integers
    // result: result
    // playerScores: playerScores
    // scores: scores
    // roll: roll
    // lastIndex: lastIndex
    
    public static List<Integer> SumScores(String[] games)//Correct
    {
        List<Integer> result = new ArrayList<Integer>();
        for(String playerScores : games)
        {
            result.add(0);
            String[] scores = playerScores.split(",",-1);
            for(String roll : scores)
            {
                int lastIndex = result.size() - 1;
                games[lastIndex] += Integer.parseInt(roll);//IDK ab this
            }
        }
        return result;
    }

    static void Main()
    {
        //#if DebugConfig
	System.out.println("WE ARE IN THE DEBUG CONFIGURATION");

        //#endif
	System.out.println("Hello, World!");
    }
}
