    public static String shortenText513(String textValue, Control control)
    {
    GraphicsContext gc = new GraphicsContext(control);
    int maxExtent = gc.textExtent(textValue).x;
    int maxWidth = control.getBounds().width - 5;
    int length = textValue.length();
    int start = length/2;
    int end = length/2 + 1;
    while (start >= 0 && end < length) {
    String s1 = textValue.substring(0, start);
    String s2 = textValue.substring(end, length);
    String s = s1 + ELLIPSIS + s2;
    int l = gc.textExtent(s).x;
    if (l < maxWidth) {
    gc.dispose();
    return s;
    }
    start--;
    end++;
    }
    return null; // Had to be added to allow compilation
    }
