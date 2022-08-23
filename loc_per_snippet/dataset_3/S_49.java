    //SNIPPET_STARTS
    public void actionPerformed(ActionEvent evt) {
        if (!hasFocus()) {
            stopBlinking();
        }

        if (blinkOn) {
            setOpaque(false);
            blinkOn = false;
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s51
