package snippet_splitter_out.ds_9$gc;
public class ds_9$gc_snip_7_getMove411 {
// SNIPPET_END_3
// Snippet 4
// raptor.chess.pgn.PgnUtils.getMove
// http://code.google.com/p/raptor‐chess‐interface/source/browse/tags/v.92/src/raptor/chess/pgn/PgnUtils.jav
// a
// S4_1:1 method chains, good comments
/**
 * Generates a string (“builder”) for a given chess move in PGN (Portable
 * Game Notation). This includes the move number and all NAG annotations.
 */
// SNIPPET_STARTS_1
public static boolean getMove411(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    /* Add move number and move details. */
    builder.append(moveNumber).append(move.isWhitesMove() ? ". " : "... ").append(move.toString());
    /* Add all available analysis data (sublines). */
    for (SublineNode subline : move.getSublines()) {
        result = true;
        builder.append(" (");
        appendSubline(builder, subline);
        builder.append(")");
    }
    /* Add all text comments of the move. */
    for (Comment comment : move.getComments()) {
        builder.append(" {").append(comment.getText()).append("}");
    }
    /* Add all Numeric Annotations Glyphs (NAGs) of the move. */
    for (Nag nag : move.getNags()) {
        builder.append(" ").append(nag.getNagString());
    }
    /* Add time taken for the move. */
    builder.append(" {").append(move.getTimeTakenForMove().getText()).append("}");
    return result;
}
}