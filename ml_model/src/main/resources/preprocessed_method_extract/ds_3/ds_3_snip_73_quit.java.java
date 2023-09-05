// Snippet s5
/**
 * Quits the application without any questions.
 */
// SNIPPET_STARTS
public void quit() {
    getConnectController().quitGame(true);
    if (!windowed) {
        gd.setFullScreenWindow(null);
    }
    System.exit(0);
}