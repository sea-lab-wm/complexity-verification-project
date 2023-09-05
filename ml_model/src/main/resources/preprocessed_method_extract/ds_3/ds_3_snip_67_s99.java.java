// Added to allow compilation
// Snippet s99
// SNIPPET_STARTS
public Object s99() {
    if (expression.exprType != VALUE && expression.exprType != COLUMN && expression.exprType != FUNCTION && expression.exprType != ALTERNATIVE && expression.exprType != CASEWHEN && expression.exprType != CONVERT) {
        StringBuffer temp = new StringBuffer();
        ddl = temp.append('(').append(ddl).append(')').toString();
    }
    return ddl;
}