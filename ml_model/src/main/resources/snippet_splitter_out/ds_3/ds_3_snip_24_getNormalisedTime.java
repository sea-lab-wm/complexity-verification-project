package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_24_getNormalisedTime {
// Added to allow compilation
// Snippet s24
// SNIPPET_STARTS
public static long getNormalisedTime(long t) {
    synchronized (tempCalDefault) {
        setTimeInMillis(tempCalDefault, t);
        resetToTime(tempCalDefault);
        return getTimeInMillis(tempCalDefault);
    }
    // Added to allow compilation
}
}