    //SNIPPET_STARTS
    public void s95() {
        InGameInputHandler inGameInputHandler = freeColClient.getInGameInputHandler();

        freeColClient.getClient().setMessageHandler(inGameInputHandler);
        gui.setInGame(true);
    } // Added to allow compilation

    // Snippet s96
    /**
     * Applies this action.
     *
     * @param e The <code>ActionEvent</code>.
     */
