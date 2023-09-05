// Snippet s16
// SNIPPET_STARTS
public void s16() {
    if (parent != null)
        setStrictJava(parent.getStrictJava());
    this.sourceFileInfo = sourceFileInfo;
    BshClassManager bcm = BshClassManager.createClassManager(this);
}