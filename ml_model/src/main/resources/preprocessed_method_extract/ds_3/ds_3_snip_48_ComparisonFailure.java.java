// SNIPPET_STARTS
public ComparisonFailure(String message, String expected, String actual) {
    super(message);
    fExpected = expected;
    fActual = actual;
}