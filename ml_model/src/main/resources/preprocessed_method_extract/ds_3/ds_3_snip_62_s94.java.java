// Added to allow compilation
// Snippet s94
// SNIPPET_STARTS
public void s94() {
    Node r = x.getRight();
    if (r != null) {
        x = r;
        Node l = x.getLeft();
    }
}