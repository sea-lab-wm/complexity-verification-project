// had to be added to allow compilation
// Snippet s6
// SNIPPET_STARTS
private boolean s6() {
    xsp = jj_scanpos;
    if (jj_scan_token(100)) {
        jj_scanpos = xsp;
        if (jj_scan_token(101))
            return true;
    }
    // had to be added to allow compilation
    // had to be added to allow compilation
    return true;
}