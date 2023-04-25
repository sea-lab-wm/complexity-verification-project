package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_87_correctTimeZone {
// Added to allow compilation
// Snippet s87
// SNIPPET_STARTS
private static Date correctTimeZone(final Date date) {
    Date ret = date;
    if (java.util.TimeZone.getDefault().useDaylightTime()) {
        if (java.util.TimeZone.getDefault().inDaylightTime(date))
            ret.setTime(date.getTime() + 1 * 60 * 60 * 1000);
    }
    return ret;
}
}