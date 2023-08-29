using System.Collections.Generic;

public class CodeStructureLbr
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
}