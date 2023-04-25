package FeatureExtraction.snippet_splitter_out;
public class ds_9$nc_snip_8_getMove {
// SNIPPET_END_2
// S4_2:3 resolved method chains, no comments
/**
 * Generates a string (“builder”) for a given chess move in PGN (Portable
 * Game Notation). This includes the move number and all NAG annotations.
 */
// SNIPPET_STARTS_3
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
}