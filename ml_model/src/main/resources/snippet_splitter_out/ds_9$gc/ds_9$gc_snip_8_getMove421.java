package FeatureExtraction.snippet_splitter_out;
public class ds_9$gc_snip_8_getMove421 {
// SNIPPET_END_3
// S4_2:1 resolved method chains, good comments
/**
 * Generates a string (“builder”) for a given chess move in PGN (Portable
 * Game Notation). This includes the move number and all NAG annotations.
 */
// SNIPPET_STARTS_1
public static boolean getMove421(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    /* Add move number and move details. */
    builder.append(moveNumber);
    builder.append(move.isWhitesMove() ? ". " : "... ");
    builder.append(move.toString());
    /* Add all available analysis data (sublines). */
    for (SublineNode subline : move.getSublines()) {
        result = true;
        builder.append(" (");
        appendSubline(builder, subline);
        builder.append(")");
    }
    /* Add all text comments of the move. */
    for (Comment comment : move.getComments()) {
        builder.append(" {");
        builder.append(comment.getText());
        builder.append("}");
    }
    /* Add all Numeric Annotations Glyphs (NAGs) of the move. */
    for (Nag nag : move.getNags()) {
        builder.append(" ");
        builder.append(nag.getNagString());
    }
    /* Add time taken for the move. */
    builder.append(" {");
    TimeTakenForMove timeTaken = move.getTimeTakenForMove();
    builder.append(timeTaken.getText());
    builder.append("}");
    return result;
}
}