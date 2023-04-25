package FeatureExtraction.snippet_splitter_out;
public class ds_9$bc_snip_9_shortenText512 {
// SNIPPET_END_1
// S5_1:2 method chains, bad comments
/**
 * Shortens the given text textValue so that its width in pixels does
 * not exceed the width of the given control. To do that, shortenText
 * overrides as many as necessary characters in the center of the original
 * text with an ellipsis character (constant ELLIPSIS = "...").
 */
// SNIPPET_STARTS_2
public static String shortenText512(String textValue, Control control) {
    /* Define local variables. */
    GraphicsContext gc = new GraphicsContext(control);
    int maxExtent = gc.textExtent(textValue).x;
    int maxWidth = control.getBounds().width - 5;
    /* Define further local variables. */
    int length = textValue.length();
    int start = length / 2;
    int end = length / 2 + 1;
    /* While the condition holds true, */
    /* override characters between s1 and s2 with ELLIPSIS. */
    while (start >= 0 && end < length) {
        String s1 = textValue.substring(0, start);
        String s2 = textValue.substring(end, length);
        String s = s1 + ELLIPSIS + s2;
        int l = gc.textExtent(s).x;
        /* When l is smaller than maxWidth, return s. */
        if (l < maxWidth) {
            gc.dispose();
            return s;
        }
        start--;
        end++;
    }
    // Had to be added to allow compilation
    return null;
}
}