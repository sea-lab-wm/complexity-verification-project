package edu.wm.kobifeldman.fMRI_Study_Classes;

import org.checkerframework.checker.nullness.qual.*;

public class YesNo {
    public static void main() {
        String input = "Yes";
        System.out.print(compute(input));
    }

    static @Nullable Boolean compute(String input) {
        input = input.toLowerCase();

        if (input.contentEquals("n")) {
            return false;
        } else if (input.contentEquals("no")) {
            return false;
        }

        if (input.contentEquals("y")) {
            return true;
        } else if (input.contentEquals("yes")) {
            return true;
        }

        return null;
    }
}