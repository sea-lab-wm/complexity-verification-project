package FeatureExtraction.snippet_splitter_out;

public class ComparisonFailure {
  // SNIPPET_STARTS
  public ComparisonFailure(String message, String expected, String actual) {
    super(message);
    fExpected = expected;
    fActual = actual;
  } // Added to allow compilation
}
