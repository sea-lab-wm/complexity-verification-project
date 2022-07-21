import fMRI_Study_Classes.*;

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

        cog_complexity_validation_datasets.One.Tasks.runAllSnippets();

        cog_complexity_validation_datasets.Three.Tasks t3 = new cog_complexity_validation_datasets.Three.Tasks("message");
        t3.runAllSnippets();
    }
}
