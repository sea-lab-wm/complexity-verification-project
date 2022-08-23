    //SNIPPET_STARTS
    public static String shortenText511(String textValue, Control control)
    {
    GraphicsContext gc = new GraphicsContext(control);
    int maxExtent = gc.textExtent(textValue).x;
    int maxWidth = control.getBounds().width - 5;
    /* Set start and end points for the center of the text. */
    int length = textValue.length();
    int start = length/2;
    int end = length/2 + 1;
    /* Take away more and more characters in the center of textValue */
    /* and replace them by a single ellipsis character until it is */
    /* short enough. */
    while (start >= 0 && end < length) {
    String s1 = textValue.substring(0, start);
    String s2 = textValue.substring(end, length);
    String s = s1 + ELLIPSIS + s2;
    int l = gc.textExtent(s).x;
    /* When the text fits, we stop and return the shortened string. */
    if (l < maxWidth) {
    gc.dispose();
    return s;
    }
    start--;
    end++;
    }
    return null; // Had to be added to allow compilation
    }

    // S5_1:2 method chains, bad comments
    /**
    * Shortens the given text textValue so that its width in pixels does
    * not exceed the width of the given control. To do that, shortenText
    * overrides as many as necessary characters in the center of the original
    * text with an ellipsis character (constant ELLIPSIS = "...").
    */
