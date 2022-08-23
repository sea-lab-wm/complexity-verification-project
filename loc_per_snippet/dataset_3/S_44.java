    //SNIPPET_STARTS
    public void s45() {
        final String heightText = Messages.message("height");

        final JTextField inputWidth = new JTextField(Integer.toString(DEFAULT_WIDTH), COLUMNS);
        final JTextField inputHeight = new JTextField(Integer.toString(DEFAULT_HEIGHT), COLUMNS);
    } // Added to allow compilation

    // Snippet s46
    /**
     Get the top level namespace or this namespace if we are the top.
     Note: this method should probably return type bsh.This to be consistent
     with getThis();
     */
