    //SNIPPET_STARTS
    public void s69() throws SQLException {
        t.checkColumnsMatch(tc.core.mainColArray, tc.core.refTable,
                tc.core.refColArray);
        session.commit();

        TableWorks tableWorks = new TableWorks(session, t);
    } // Added to allow compilation

    // Snippet s70
//    @Override // Removed to allow compilation
