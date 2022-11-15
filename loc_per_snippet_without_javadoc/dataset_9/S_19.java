    public static boolean getMove412(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    /* Add move data. */
    builder.append(moveNumber).append(move.isWhitesMove() ? ". " : "... ").
    append(move.toString());
    /* Add sublines. */
    for (SublineNode subline : move.getSublines()) {
    result = true;
    builder.append(" (");
    appendSubline(builder, subline);
    builder.append(")");
    }
    /* Add comments. */
    for (Comment comment : move.getComments()) {
    builder.append(" {").append(comment.getText()).append("}");
    }
    /* Add NAGs. */
    for (Nag nag : move.getNags()) {
    builder.append(" ").append(nag.getNagString());
    }
    /* Add time. */
    builder.append(" {").append(move.getTimeTakenForMove().getText()).append("}");
    return result;
    }
