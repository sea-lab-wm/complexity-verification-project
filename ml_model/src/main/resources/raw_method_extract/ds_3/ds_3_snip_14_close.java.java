// Snippet s49
// SNIPPET_STARTS
public void close() {
    if (isClosed) {
        return;
    }
    isClosed = true;
    try {
        resultOut.setResultType(ResultConstants.SQLDISCONNECT);
    } finally {
        // Added to allow compilation
    }
}