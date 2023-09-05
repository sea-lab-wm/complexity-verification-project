// Added to allow compilation
// Snippet s84
// SNIPPET_STARTS
public void s84() {
    /* fredt - in FK constraints column lists for iColMain and iColRef have
           identical sets to visible columns of iMain and iRef respectively
           but the order of columns can be different and must be preserved
         */
    core.mainColArray = mainCols;
    core.colLen = core.mainColArray.length;
    core.refColArray = refCols;
}