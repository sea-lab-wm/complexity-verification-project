    //SNIPPET_STARTS
    public static boolean getMove(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    builder.append(moveNumber);
    builder.append(move.isWhitesMove() ? ". " : "... ");
    builder.append(move.toString());
    for (SublineNode subline : move.getSublines()) {
    result = true;
    builder.append(" (");
    appendSubline(builder, subline);
    builder.append(")");
    }
    for (Comment comment : move.getComments()) {
    builder.append(" {");
    builder.append(comment.getText());
    builder.append("}");
    }
    for (Nag nag : move.getNags()) {
    builder.append(" ");
    builder.append(nag.getNagString());
    }
    builder.append(" {");
    TimeTakenForMove timeTaken = move.getTimeTakenForMove();
    builder.append(timeTaken.getText());
    builder.append("}");
    return result;
    }

    // Snippet 5
    // org.eclipse.jface.dialogs.Dialog.shortenText
    // http://www.docjar.com/html/api/org/eclipse/jface/dialogs/Dialog.java.html

    // S5_1:1 method chains, good comments
    /**
    * Shortens the given text textValue so that its width in pixels does
    * not exceed the width of the given control. To do that, shortenText
    * overrides as many as necessary characters in the center of the original
    * text with an ellipsis character (constant ELLIPSIS = "...").
    */
