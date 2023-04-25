package FeatureExtraction.snippet_splitter_out;
public class ds_f_snip_RecursiveFactorial_compute {
// SNIPPET_STARTS
public static int compute(int value) {
    if (value == 1) {
        return 1;
    }
    return compute(value - 1) * value;
}
}