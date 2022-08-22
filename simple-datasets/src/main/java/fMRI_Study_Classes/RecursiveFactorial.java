package fMRI_Study_Classes;

import gov.nasa.jpf.vm.Verify;

public class RecursiveFactorial {
    public static void run() {
        int input = 4;
        //int input = Verify.getInt(-128, 127);
        System.out.print(compute(input));
    }

    public static int compute(int value) {
        if (value == 1) {
            return 1;
        }

        return compute(value - 1) * value;
    }
}