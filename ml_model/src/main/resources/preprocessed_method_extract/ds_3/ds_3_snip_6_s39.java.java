// Snippet s39
// SNIPPET_STARTS
public void s39() {
    row[1] = ns.getCatalogName(row[0]);
    row[2] = schema.equals(defschema) ? Boolean.TRUE : Boolean.FALSE;
    t.insertSys(row);
}