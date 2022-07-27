import fMRI_Study_Classes.*;

public class Main {
    public static void main(String[] args) {
        //String s = new String("Abc");
        //String u = "bc";
        //assert(!s.contains(u)); // This assert failure should be found by JBMC!

        //Object n = null;
        //n.toString();

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

        cog_complexity_validation_datasets.Three.Tasks t3 = new cog_complexity_validation_datasets.Three.Tasks("message");
        t3.runAllSnippets();
    }
}
