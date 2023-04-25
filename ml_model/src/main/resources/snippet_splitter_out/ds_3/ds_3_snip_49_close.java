package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_49_close {
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
}