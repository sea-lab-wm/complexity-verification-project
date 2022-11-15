    /**
    * Generates a string (“builder”) for a given chess move in PGN (Portable
    * Game Notation). This includes the move number and all NAG annotations.
    */
    public static boolean getMove413(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    builder.append(moveNumber).append(move.isWhitesMove() ? ". " : "... ").
    append(move.toString());
    for (SublineNode subline : move.getSublines()) {
    result = true;
    builder.append(" (");
    appendSubline(builder, subline);
    builder.append(")");
    }
    for (Comment comment : move.getComments()) {
    builder.append(" {").append(comment.getText()).append("}");
    }
    for (Nag nag : move.getNags()) {
    builder.append(" ").append(nag.getNagString());
    }
    builder.append(" {").append(move.getTimeTakenForMove().getText()).append("}");
    return result;
    }
