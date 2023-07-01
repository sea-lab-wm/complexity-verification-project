package FeatureExtraction.snippet_splitter_out;
public class TestClassRunnerForParameters {
    //SNIPPET_STARTS
    private TestClassRunnerForParameters(Class<?> klass, Object[] parameters, int i) {
        super(klass);
        fParameters= parameters;
        fParameterSetNumber= i;
    } // Added to allow compilation
}
