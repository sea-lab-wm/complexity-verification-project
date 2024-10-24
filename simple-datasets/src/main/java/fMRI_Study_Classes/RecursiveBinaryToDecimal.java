package fMRI_Study_Classes;

public class RecursiveBinaryToDecimal {
    public static void run() {
        String input = "101";
        int number = 1;
        System.out.print(compute(input, number));
    }

    //SNIPPET_STARTS
    static int compute(String s, int number) {
        if (number < 0) {
            return 0;
        }

        if (s.charAt(number) == '0'){
            return 2 * compute(s, number - 1);
        }

        return 1 + 2 * compute(s, number - 1);
    }
}