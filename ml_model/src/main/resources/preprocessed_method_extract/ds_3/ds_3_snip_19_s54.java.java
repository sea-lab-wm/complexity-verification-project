// Added to allow compilation
// Snippet s54
// SNIPPET_STARTS
public void s54() {
    // ----------------------------------------------------------------
    // required
    // ----------------------------------------------------------------
    addColumn(t, "PROCEDURE_CAT", Types.VARCHAR);
    addColumn(t, "PROCEDURE_SCHEM", Types.VARCHAR);
    // not null
    addColumn(t, "PROCEDURE_NAME", Types.VARCHAR, false);
}