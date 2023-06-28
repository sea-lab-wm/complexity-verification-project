package snippet_splitter_out.ds_9$gc;
public class ds_9$gc_snip_10_shortenText521 {
// SNIPPET_END_3
// S5_2:1 resolved method chains, good comments
/**
 * Shortens the given text textValue so that its width in pixels does
 * not exceed the width of the given control. To do that, shortenText
 * overrides as many as necessary characters in the center of the original
 * text with an ellipsis character (constant ELLIPSIS = "...").
 */
// SNIPPET_STARTS_1
public static String shortenText521(String textValue, Control control) {
    GraphicsContext gc = new GraphicsContext(control);
    Extent extent = gc.textExtent(textValue);
    int maxExtent = extent.x;
    Bounds bounds = control.getBounds();
    int maxWidth = bounds.width - 5;
    /* Set start and end points for the center of the text. */
    int length = textValue.length();
    int start = length / 2;
    int end = length / 2 + 1;
    /* Take away more and more characters in the center of textValue */
    /* and replace them by a single ellipsis character until it is */
    /* short enough. */
    while (start >= 0 && end < length) {
        String s1 = textValue.substring(0, start);
        String s2 = textValue.substring(end, length);
        String s = s1 + ELLIPSIS + s2;
        extent = gc.textExtent(s);
        int l = extent.x;
        /* When the text fits, we stop and return the shortened string. */
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