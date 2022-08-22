package fMRI_Study_Classes;

import java.util.Arrays;

import gov.nasa.jpf.vm.Verify;

public class SquareRoot {
    public static void run() {
        int[] numbers = {9, 25, 16, 100};
        //int[] numbers = {Verify.getInt(-128, 127)};
        System.out.print(compute(numbers));
    }

    public static String compute(int[] numbers) {
        double[] result = new double[numbers.length];

        for (int i = 0; i < numbers.length; i++) {
            if (numbers[i] == 0) {
                result[i] = 0;
                continue;
            }

            if (numbers[i] < 0) {
                result[i] = Math.sqrt(-1 * numbers[i]);
            } else {
                result[i] = Math.sqrt(numbers[i]);
            }
        }

        return Arrays.toString(result);
    }
}