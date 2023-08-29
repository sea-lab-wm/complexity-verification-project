using System.Collections.Generic;
public class MergelistsSemanticNormal
{
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
}