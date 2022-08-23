    //SNIPPET_STARTS
    public void s64() {
        while (classNames.hasNext()) {
            clsName = (String) classNames.next();
            clsCat = ns.getCatalogName(clsName);
            clsSchem = ns.getSchemaName(clsName);
        }
    } // Added to allow compilation

    // Snippet s65
