// Added to allow compilation
// Snippet s79
// SNIPPET_STARTS
public int s79() {
    for (int j = 0; j < fieldcount; j++) {
        int i = Column.compare(session.database.collation, a[cols[j]], b[cols[j]], coltypes[cols[j]]);
        if (i != 0) {
            return i;
        }
    }
    return 0;
}