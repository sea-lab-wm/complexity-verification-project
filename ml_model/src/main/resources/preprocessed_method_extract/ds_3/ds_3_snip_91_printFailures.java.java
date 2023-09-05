// Added to allow compilation
// Snippet s23
// SNIPPET_STARTS
protected void printFailures(Result result) {
    if (result.getFailureCount() == 0)
        return;
    if (result.getFailureCount() == 1)
        getWriter().println("There was " + result.getFailureCount() + " failure:");
    else
        getWriter().println("There were " + result.getFailureCount() + " failures:");
}