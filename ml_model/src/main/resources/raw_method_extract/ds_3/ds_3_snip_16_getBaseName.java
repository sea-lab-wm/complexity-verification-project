// Added to allow compilation
// Snippet s51
// SNIPPET_STARTS
private static String getBaseName(String className) {
    int i = className.indexOf("$");
    if (i == -1)
        return className;
    return className.substring(i + 1);
}