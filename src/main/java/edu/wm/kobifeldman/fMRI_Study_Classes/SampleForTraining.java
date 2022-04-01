package edu.wm.kobifeldman.fMRI_Study_Classes;

public class SampleForTraining {
    public static void main() {
        int number = 13;
        System.out.print(compute(number));
    }

    public static boolean compute(int number) {
        if (number < 5) {
            return false;
        } else {
            return true;
        }
    }
}