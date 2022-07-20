import fMRI_Study_Classes.*;
//import cog_complexity_validation_datasets.One.*;
import cog_complexity_validation_datasets.Three.*;

public class Main {
    public static void main(String[] args) {
        ArrayAverage.main();
        ContainsSubstring.main();
        CountVowels.main();
        DumbSort.main();
        GreatestCommonDivisor.main();
        hIndex.main();
        isHurricane.main();
        isPalindrome.main();
        lengthOfLastWord.main();
        RecursiveBinaryToDecimal.main();
        RecursiveCrossSum.main();
        RecursiveFactorial.main();
        RecursiveFibonacciVariant.main();
        RecursivePower.main();
        SquareRoot.main();
        YesNo.main();

        Tasks t3 = new Tasks("message");
        t3.runAllSnippets();
    }
}
