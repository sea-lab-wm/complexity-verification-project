package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_68_run {
// Added to allow compilation
// Snippet s68
// @Override // Removed to allow compilation
// SNIPPET_STARTS
public void run(RunNotifier notifier) {
    TestResult result = new TestResult();
    result.addListener(createAdaptingListener(notifier));
    fTest.run(result);
}
}