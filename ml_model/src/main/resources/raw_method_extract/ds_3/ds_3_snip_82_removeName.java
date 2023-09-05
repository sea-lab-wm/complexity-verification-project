// added to allow compilation
// Snippet s14
// SNIPPET_STARTS
Object removeName(String name) throws HsqlException {
    Object owner = nameList.remove(name);
    if (owner == null) {
        // should contain name
        throw Trace.error(Trace.GENERAL_ERROR);
    }
    return owner;
}