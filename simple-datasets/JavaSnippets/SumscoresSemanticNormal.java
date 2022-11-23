//Fix this line: using System.Collections.Generic;
import java.util.*;
import java.io.*;
public class SumscoresSemanticNormal
{
    // SumScores: Sums the scores of a collection of bowling games per player
    // games: collection of scores from bowling games as comma separated integers
    // result: result
    // playerScores: playerScores
    // scores: scores
    // roll: roll
    // lastIndex: lastIndex

    public static List<Integer> SumScores(String[] games)//dont know ab this 
    {
        List<Integer> result = new ArrayList<Integer>();
        for(String playerScores: games)
        {
            result.add(0);
            String[] scores = playerScores.split(",", -1);
            for(String roll: scores)
            {
                int lastIndex = result.size() - 1;
                games[lastIndex] += Integer.parseInt(roll);// dont know ab this line
            }
        }
        return result;
    }
}
