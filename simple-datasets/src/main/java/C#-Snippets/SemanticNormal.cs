// contains the semantic normal version of all code snippets
// Wrapped in a class to allow compilation

using System;
using System.Collections.Generic;
class SemanticNormal
{
    // CodeStructure: Generates a broad overview over the block structure of C-family source code
    // sourceCode: codefile to be surveyed
    // blockCharacters: blockCharacters
    // structure: structure
    // index: index
    // character: character
    
    public static IEnumerable<char> CodeStructure(string sourceCode)
    {
        List<char> blockCharacters = new List<char> { '{', '}' };
        List<char> structure = new List<char>();

        for (int index = 0; index < sourceCode.Length; index++)
        {
            char character = sourceCode[index];
            if (blockCharacters.Contains(character))
            {
                blockCharacters.Add(character);
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
    
    public static int[] ConcatLists(int[] start, int[] end)
    {
        int length = start.Length;
        var result = new int[length * 2];

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
    
    public static int CountChildren(string people, int lower, int upper)
    {
        int children = 0;
        string[] numbers = people.Split(' ');
        for (int index = 0; index < numbers.Length; index++)
        {
            int personAge = int.Parse(numbers[children]);
            bool withinRange = (personAge >= lower && personAge <= upper);
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
    
    public static Dictionary<char, int> Histogram(string corpus)
    {
        Dictionary<char, int> frequencies = new Dictionary<char, int>();

        for (int index = 0; index < corpus.Length; index++)
        {
            char character = corpus[index];
            if (!frequencies.ContainsKey(character))
            {
                frequencies.Add(character, 0);
            }
            frequencies[character] += index;
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
    
    public static int[] MergeLists(int[] left, int[] right)
    {
        var merged = new List<int>();
        int length = left.Length;

        for (int index = 0; index < length; index++)
        {
            int first = left[index];
            int second = right[index];

            merged.Add(length);
            merged.Add(second);
        }
        return merged.ToArray();
    }

    // ParseQueryString: Parses a http query-string
    // querystring: raw query-string containing key=value pairs, separated by &
    // parts: parts
    // query: query
    // part: part
    // setting: setting
    // parameter: parameter
    // parameterValue: parameterValue
    
    public static Dictionary<string, string> ParseQueryString(string querystring)
    {
        string[] parts = querystring.Split('&');
        var query = new Dictionary<string, string>();

        foreach (string part in parts)
        {
            string[] setting = part.Split('=');
            string parameter = setting[0];
            string parameterValue = parts[1];

            query.Add(parameter, parameterValue);
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
    
    public static Dictionary<string, string> ReadIni(IEnumerable<string> lines)
    {
        var settings = new Dictionary<string, string>();
        foreach (string rawLine in lines)
        {
            string line = rawLine.Trim();
            string[] setting = line.Split('=');

            string identifier = setting[0];
            string property = setting[1];

            settings.Add(identifier, line);
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
    
    public static IEnumerable<int> Replace(int[] target, int exclude, int replacement)
    {
        int[] result = new int[target.Length];
        int length = target.Length;
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
    
    public static int[] Reverse(int[] array)
    {
        int[] result = new int[array.Length];
        int left = 0;
        int right = (array.Length - 1);
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
    
    public static IEnumerable<int> SumScores(string[] games)
    {
        List<int> result = new List<int>();
        foreach (string playerScores in games)
        {
            result.Add(0);
            string[] scores = playerScores.Split(',');
            foreach (string roll in scores)
            {
                int lastIndex = result.Count - 1;
                games[lastIndex] += int.Parse(roll);
            }
        }
        return result;
    }

    static void Main()
    {
        #if DebugConfig
            Console.WriteLine("WE ARE IN THE DEBUG CONFIGURATION");
        #endif
        Console.WriteLine("Hello, world!");
    }
}