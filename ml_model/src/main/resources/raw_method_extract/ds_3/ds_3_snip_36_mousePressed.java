// Added to allow compilation
// Snippet s70
// @Override // Removed to allow compilation
// SNIPPET_STARTS
public void mousePressed(MouseEvent e) {
    if (f.getDesktopPane() == null || f.getDesktopPane().getDesktopManager() == null) {
        return;
    }
    loc = SwingUtilities.convertPoint((Component) e.getSource(), e.getX(), e.getY(), null);
    f.getDesktopPane().getDesktopManager().beginDraggingFrame(f);
}