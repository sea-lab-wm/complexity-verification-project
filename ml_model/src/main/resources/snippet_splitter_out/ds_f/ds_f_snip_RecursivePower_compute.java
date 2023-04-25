package FeatureExtraction.snippet_splitter_out;
public class ds_f_snip_RecursivePower_compute {
// SNIPPET_STARTS
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