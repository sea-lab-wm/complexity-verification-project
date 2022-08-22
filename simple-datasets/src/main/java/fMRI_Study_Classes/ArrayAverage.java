package fMRI_Study_Classes;

import gov.nasa.jpf.vm.Verify;

public class ArrayAverage {
    public static void run() {
        int[] numbers = {2, 4, 1, 9};
        //int[] numbers = {-128, 0, Verify.getInt(0, 10), Verify.getInt(0, 10)};
        System.out.print(compute(numbers));
    }

    public static float compute(int[] numbers) {
        int number1 = 0;
        int number2 = 0;

        while (number1 < numbers.length) {
            number2 = number2 + numbers[number1];
            number1 = number1 + 1;
        }

        float result = number2 / (float) number1;
        return result;
    }
}