package fMRI_Study_Classes;

import java.util.Arrays;
import java.util.List;

import gov.nasa.jpf.vm.Verify;

public class DumbSort {
    public static void run() {
        //int a = 9;
        //int b = 12;
        //int c = 8;
        //int d = 11;
        int a = Verify.getInt(-1, 7);
        int b = Verify.getInt(-1, 7);
        int c = Verify.getInt(-1, 7);
        int d = Verify.getInt(-1, 7);
        System.out.print(compute(a, b, c, d));
    }

    public static List<Integer> compute(int a, int b, int c, int d) {
        if (a > b) { int temp = b; b = a; a = temp; }
        if (c > d) { int temp = d; d = c; c = temp; }
        if (a > c) { int temp = c; c = a; a = temp; }
        if (b > d) { int temp = d; d = b; b = temp; }
        if (b > c) { int temp = c; c = b; b = temp; }

        return Arrays.asList(a, b, c, d);
    }
}