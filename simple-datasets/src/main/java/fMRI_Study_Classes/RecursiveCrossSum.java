package fMRI_Study_Classes;

public class RecursiveCrossSum {
    public static void run() {
        int n = 3247;
        System.out.print(compute(n));
    }

    //SNIPPET_STARTS
    public static int compute(int number) {
        if (number == 0) {
            return 0;
        }

        return (number % 10) + compute((int) number/10);
    }
}