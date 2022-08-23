    //SNIPPET_STARTS
    public void quit() {
        getConnectController().quitGame(true);
        if (!windowed) {
            gd.setFullScreenWindow(null);
        }
        System.exit(0);
    }   // had to be added to allow compilation

    // Snippet s6
