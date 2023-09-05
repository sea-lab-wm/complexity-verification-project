// Snippet s3
// SNIPPET_STARTS
private String compactString(String source) {
    String result = DELTA_START + source.substring(fPrefix, source.length() - fSuffix + 1) + DELTA_END;
    if (fPrefix > 0)
        result = computeCommonPrefix() + result;
    if (fSuffix > 0)
        result = result + computeCommonSuffix();
    // had to be added to allow compilation
    return result;
}