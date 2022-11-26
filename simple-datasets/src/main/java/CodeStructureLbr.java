//using System.Collections.Generic;
import java.io.*;
import java.util.*;
import java.util.ArrayList;
public class CodeStructureLbr
{
    // CodeStructure: Generates a broad overview over the block structure of C-family source code
    // sourceCode: codefile to be surveyed
    // blockCharacters: blockCharacters
    // structure: structure
    // index: index
    // character: character

    public static List<Character> CodeStructure(String sourceCode)//check this IEnumerable
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
}
