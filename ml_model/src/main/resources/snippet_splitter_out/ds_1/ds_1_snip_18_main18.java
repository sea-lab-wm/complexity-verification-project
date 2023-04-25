package FeatureExtraction.snippet_splitter_out;
public class ds_1_snip_18_main18 {
// DATASET2END
// 18. Least common multiple                                /*Only in the first pilot study*/
// SNIPPET_STARTS
public static void main18(String[] args) {
    int number1 = 23;
    int number2 = 42;
    int max, min;
    // Note: a ";" had to be added here to allow compilation
    int results = -1;
    if (number1 > number2) {
        max = number1;
        min = number2;
    } else {
        max = number2;
        min = number1;
    }
    for (int i = 1; i <= min; i++) {
        if ((max * i) % min == 0) {
            // Note: result had to be renamed to results to allow compilation
            results = i * max;
            // Note: result had to be renamed to results to allow compilation
            break;
        }
    }
    if (// Note: result had to be renamed to results to allow compilation
    results != -1)
        System.out.println(results);
    else
        System.out.println("Error!");
}
}