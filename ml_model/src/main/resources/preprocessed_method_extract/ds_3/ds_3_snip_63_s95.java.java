// Added to allow compilation
// Snippet s95
// SNIPPET_STARTS
public void s95() {
    InGameInputHandler inGameInputHandler = freeColClient.getInGameInputHandler();
    freeColClient.getClient().setMessageHandler(inGameInputHandler);
    gui.setInGame(true);
}