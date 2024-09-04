package snippet_splitter_out.ds_9$nc;
public class ds_9$nc_snip_10_shortenText523 {
// SNIPPET_END_2
// S5_2:3 resolved method chains, no comments
/**
 * Shortens the given text textValue so that its width in pixels does
 * not exceed the width of the given control. To do that, shortenText
 * overrides as many as necessary characters in the center of the original
 * text with an ellipsis character (constant ELLIPSIS = "...").
 */
// SNIPPET_STARTS_3
public static String shortenText523(String textValue, Control control) {
    GraphicsContext gc = new GraphicsContext(control);
    Extent extent = gc.textExtent(textValue);
    int maxExtent = extent.x;
    Bounds bounds = control.getBounds();
    int maxWidth = bounds.width - 5;
    int length = textValue.length();
    int start = length / 2;
    int end = length / 2 + 1;
    while (start >= 0 && end < length) {
        String s1 = textValue.substring(0, start);
        String s2 = textValue.substring(end, length);
        String s = s1 + ELLIPSIS + s2;
        extent = gc.textExtent(s);
        int l = extent.x;
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