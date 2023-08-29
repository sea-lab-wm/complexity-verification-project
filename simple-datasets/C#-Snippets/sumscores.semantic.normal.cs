using System.Collections.Generic;
public class SumscoresSemanticNormal
{
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
}