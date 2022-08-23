    //SNIPPET_STARTS
    public void s26() {
        if ((bufpos + 1) >= len)
            System.arraycopy(buffer, bufpos - len + 1, ret, 0, len);
        else {
            System.arraycopy(buffer, bufsize - (len - bufpos - 1), ret, 0,
                    len - bufpos - 1);
            System.arraycopy(buffer, 0, ret, len - bufpos - 1, bufpos + 1);
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s27
    /**
     * Compute the proper position for a centered window
     */
