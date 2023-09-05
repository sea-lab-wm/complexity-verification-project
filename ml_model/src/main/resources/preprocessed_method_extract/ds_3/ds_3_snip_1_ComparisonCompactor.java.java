// Snippet s33
/**
 * @param contextLength the maximum length for <code>expected</code> and <code>actual</code>. When
 *     contextLength is exceeded, the Strings are shortened
 * @param expected the expected string value
 * @param actual the actual string value
 */
// SNIPPET_STARTS
public void ComparisonCompactor(int contextLength, String expected, String actual) {
    // return type void added to allow compilation
    fContextLength = contextLength;
    fExpected = expected;
    fActual = actual;
}