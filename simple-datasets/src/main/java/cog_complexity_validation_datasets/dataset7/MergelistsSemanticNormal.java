//using System.Collections.Generic;
import java.util.*;
import java.io.*;
import java.util.ArrayList;
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
        List<Integer>  merged = new ArrayList<Integer>();
        int length = left.length;

        for (int index = 0; index < length; index++)
        {
            int first = left[index];
            int second = right[index];

            merged.add(length);
            merged.add(second);
        }
	int[] arr = new int[merged.size()];
	for(int i =0; i < merged.size(); i++) arr[i] = merged.get(i);	
        return arr;//might cause only problem
    }
}
