package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_31_s31 {
// Added to allow compilation
// Snippet s31
// SNIPPET_STARTS
public String s31() {
    if (iterateOverMe instanceof String)
        return createEnumeration(((String) iterateOverMe).toCharArray());
    if (iterateOverMe instanceof StringBuffer)
        return createEnumeration(iterateOverMe.toString().toCharArray());
    throw new IllegalArgumentException("Cannot enumerate object of type " + iterateOverMe.getClass());
}
}