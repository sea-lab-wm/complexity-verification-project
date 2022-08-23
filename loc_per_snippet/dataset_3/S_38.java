    //SNIPPET_STARTS
    public void s39() {
        row[1] = ns.getCatalogName(row[0]);
        row[2] = schema.equals(defschema) ? Boolean.TRUE
                : Boolean.FALSE;

        t.insertSys(row);
    } // Added to allow compilation

    // Snippet s40
    /**
     * Handles an "deliverGift"-request.
     *
     * @param element The element (root element in a DOM-parsed XML tree) that
     *            holds all the information.
     */
