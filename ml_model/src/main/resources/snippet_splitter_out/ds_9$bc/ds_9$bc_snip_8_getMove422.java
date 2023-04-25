package FeatureExtraction.snippet_splitter_out;
public class ds_9$bc_snip_8_getMove422 {
// SNIPPET_END_1
// S4_2:2 resolved method chains, bad comments
/**
 * Generates a string (“builder”) for a given chess move in PGN (Portable
 * Game Notation). This includes the move number and all NAG annotations.
 */
// SNIPPET_STARTS_2
public static boolean getMove422(StringBuilder builder, Move move) {
    boolean result = false;
    int moveNumber = move.getFullMoveCount();
    /* Add move data. */
    builder.append(moveNumber);
    builder.append(move.isWhitesMove() ? ". " : "... ");
    builder.append(move.toString());
    /* Add sublines. */
    for (SublineNode subline : move.getSublines()) {
        result = true;
        builder.append(" (");
        appendSubline(builder, subline);
        builder.append(")");
    }
    /* Add comments. */
    for (Comment comment : move.getComments()) {
        builder.append(" {");
        builder.append(comment.getText());
        builder.append("}");
    }
    /* Add NAGs. */
    for (Nag nag : move.getNags()) {
        builder.append(" ");
        builder.append(nag.getNagString());
    }
    /* Add time. */
    builder.append(" {");
    TimeTakenForMove timeTaken = move.getTimeTakenForMove();
    builder.append(timeTaken.getText());
    builder.append("}");
    return result;
}
}