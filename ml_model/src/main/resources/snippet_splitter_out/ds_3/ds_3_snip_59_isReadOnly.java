package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_59_isReadOnly {
// Added to allow compilation
// Snippet s59
// SNIPPET_STARTS
public boolean isReadOnly() throws HsqlException {
    Object info = getAttribute(Session.INFO_CONNECTION_READONLY);
    isReadOnly = ((Boolean) info).booleanValue();
    return isReadOnly;
}
}