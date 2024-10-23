package snippet_splitter_out.ds_9$nc;
public class ds_9$nc_snip_7_getMove413 {
// SNIPPET_END_2
// S4_1:3 method chains, no comments
/**
 * Generates a string (“builder”) for a given chess move in PGN (Portable
 * Game Notation). This includes the move number and all NAG annotations.
 */
// SNIPPET_STARTS_3
public static boolean getMove413(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    builder.append(moveNumber).append(move.isWhitesMove() ? ". " : "... ").append(move.toString());
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
}