package FeatureExtraction.snippet_splitter_out;
public class ds_f_snip_RecursiveFibonacciVariant_compute {
// SNIPPET_STARTS
public static int compute(int number) {
    if (number <= 1) {
        return 1;
    }
    return compute(number - 2) + compute(number - 4);
}
}