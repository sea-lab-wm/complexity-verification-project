// Added to allow compilation
// Snippet s50
// SNIPPET_STARTS
public void actionPerformed(ActionEvent evt) {
    if (!hasFocus()) {
        stopBlinking();
    }
    if (blinkOn) {
        setOpaque(false);
        blinkOn = false;
    }
    // Added to allow compilation
}