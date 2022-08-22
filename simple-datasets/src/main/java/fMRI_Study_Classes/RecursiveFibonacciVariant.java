package fMRI_Study_Classes;

public class RecursiveFibonacciVariant {
    public static void run() {
        int number = 4;
        System.out.print(compute(number));
    }

    //SNIPPET_STARTS
    public static int compute(int number) {
        if (number <= 1) {
            return 1;
        }

        return compute(number - 2) + compute(number - 4);
    }
}