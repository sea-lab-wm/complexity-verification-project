package fMRI_Study_Classes;

import gov.nasa.jpf.vm.Verify;

public class RecursiveFibonacciVariant {
    public static void run() {
        int number = 4;
        //int number = Verify.getInt(-128, 127);
        System.out.print(compute(number));
    }

    public static int compute(int number) {
        if (number <= 1) {
            return 1;
        }

        return compute(number - 2) + compute(number - 4);
    }
}