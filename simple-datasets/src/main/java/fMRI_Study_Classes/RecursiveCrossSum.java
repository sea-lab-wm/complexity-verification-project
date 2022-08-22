package fMRI_Study_Classes;

import gov.nasa.jpf.vm.Verify;

public class RecursiveCrossSum {
    public static void run() {
        int n = 3247;
        //int n = Verify.getIntFromList(3247, 0, 10, 11, 23455);
        System.out.print(compute(n));
    }

    public static int compute(int number) {
        if (number == 0) {
            return 0;
        }

        return (number % 10) + compute((int) number/10);
    }
}