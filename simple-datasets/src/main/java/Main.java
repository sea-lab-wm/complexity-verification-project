import fMRI_Study_Classes.*;

public class Main {
    public static void main(String[] args) {
        ArrayAverage.run();
        ContainsSubstring.run();
        CountVowels.run();
        DumbSort.run();
        GreatestCommonDivisor.run();
        hIndex.run();
        isHurricane.run();
        isPalindrome.run();
        lengthOfLastWord.run();
        RecursiveBinaryToDecimal.run();
        RecursiveCrossSum.run();
        RecursiveFactorial.run();
        RecursiveFibonacciVariant.run();
        RecursivePower.run();
        SquareRoot.run();
        YesNo.run();

        cog_complexity_validation_datasets.One.Tasks.runAllSnippets();

        cog_complexity_validation_datasets.Three.Tasks_1 t3_1 = new cog_complexity_validation_datasets.Three.Tasks_1("message");
        t3_1.runAllSnippets();
        cog_complexity_validation_datasets.Three.Tasks_2 t3_2 = new cog_complexity_validation_datasets.Three.Tasks_2("message");
        t3_2.runAllSnippets();
        cog_complexity_validation_datasets.Three.Tasks_3 t3_3 = new cog_complexity_validation_datasets.Three.Tasks_3("message");
        t3_3.runAllSnippets(); 
    }
}
