import java.util.*;
import java.io.*;
public class ReverseSemanticNormal
{
    // Reverse: Reverses the items of a collection
    // array: collection of items to reverse
    // result: result
    // left: left
    // right: right
    // auxiliary: auxiliary

    public static int[] Reverse(int[] array)
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
}
