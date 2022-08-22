package fMRI_Study_Classes;

import gov.nasa.jpf.vm.Verify;

public class RecursivePower {
    public static void run() {
        int a = 3;
        int b = 2;
        //int a = Verify.getInt(-128, 127);
        //int b = Verify.getInt(-128, 127);
        System.out.print(compute(a, b));
    }

    static int compute(int a, int b) {
        if (b == 0) {
            return 1;
        }

        if (b == 1) {
            return a;
        }

        return (a + 1) * compute(a, b - 1);
    }
}