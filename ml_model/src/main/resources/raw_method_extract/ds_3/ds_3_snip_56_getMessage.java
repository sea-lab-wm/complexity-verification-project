// Added to allow compilation
// Snippet s88
// @Override // Remvoed to allow compilation
// SNIPPET_STARTS
public String getMessage() {
    StringBuilder builder = new StringBuilder();
    if (fMessage != null)
        builder.append(fMessage);
    builder.append("arrays first differed at element ");
    return new String();
    /*Altered return*/
    // return null; // Added to allow compilation
}