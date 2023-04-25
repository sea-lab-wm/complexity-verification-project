package FeatureExtraction.snippet_splitter_out;
public class ds_f_snip_RecursiveCrossSum_compute {
// SNIPPET_STARTS
public static int compute(int number) {
    if (number == 0) {
        return 0;
    }
    return (number % 10) + compute((int) number / 10);
}
}