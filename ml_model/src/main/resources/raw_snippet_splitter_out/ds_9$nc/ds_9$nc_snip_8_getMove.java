package snippet_splitter_out.ds_9$nc;
public class ds_9$nc_snip_8_getMove {
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