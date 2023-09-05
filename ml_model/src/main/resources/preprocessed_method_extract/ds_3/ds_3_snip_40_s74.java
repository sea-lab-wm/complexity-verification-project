// Added to allow compilation
// Snippet s74
// SNIPPET_STARTS
public void s74() {
    classNames = classNameSet.iterator();
    while (classNames.hasNext()) {
        className = (String) classNames.next();
        methods = iterateRoutineMethods(className, andAliases);
    }
    // Added to allow compilation
}