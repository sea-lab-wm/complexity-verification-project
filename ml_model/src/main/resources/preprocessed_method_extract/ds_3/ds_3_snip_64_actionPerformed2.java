// Added to allow compilation
// Snippet s96
/**
 * Applies this action.
 *
 * @param e The <code>ActionEvent</code>.
 */
// SNIPPET_STARTS
public void actionPerformed2(ActionEvent e) {
    // Renamed to allow compilation
    final Game game = freeColClient.getGame();
    final Map map = game.getMap();
    Parameters p = showParametersDialog();
}