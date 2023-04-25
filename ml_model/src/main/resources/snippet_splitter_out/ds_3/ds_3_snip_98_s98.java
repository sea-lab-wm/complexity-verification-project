package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_98_s98 {
// Added to allow compilation
// Snippet s98
// SNIPPET_STARTS
public void s98() {
    Description description = Description.createSuiteDescription(name);
    int n = ts.testCount();
    for (int i = 0; i < n; i++) description.addChild(makeDescription(ts.testAt(i)));
}
}