    public void s39() {
        row[1] = ns.getCatalogName(row[0]);
        row[2] = schema.equals(defschema) ? Boolean.TRUE
                : Boolean.FALSE;

        t.insertSys(row);
    } // Added to allow compilation
