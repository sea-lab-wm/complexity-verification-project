// Added to allow compilation
// Snippet s69
// SNIPPET_STARTS
public void s69() throws SQLException {
    t.checkColumnsMatch(tc.core.mainColArray, tc.core.refTable, tc.core.refColArray);
    session.commit();
    TableWorks tableWorks = new TableWorks(session, t);
}