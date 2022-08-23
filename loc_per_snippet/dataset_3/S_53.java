    //SNIPPET_STARTS
    public void s54() {
        // ----------------------------------------------------------------
        // required
        // ----------------------------------------------------------------
        addColumn(t, "PROCEDURE_CAT", Types.VARCHAR);
        addColumn(t, "PROCEDURE_SCHEM", Types.VARCHAR);
        addColumn(t, "PROCEDURE_NAME", Types.VARCHAR, false);    // not null
    } // Added to allow compilation

    // Snippet s55
