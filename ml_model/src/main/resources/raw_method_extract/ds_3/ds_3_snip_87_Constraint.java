// Added to allow compilation
// Snippet s19
/**
 * temp constraint constructor
 */
// SNIPPET_STARTS
// Added return type void to allow compilation
// Added return type void to allow compilation
void // Added return type void to allow compilation
Constraint(// Added return type void to allow compilation
HsqlName name, // Added return type void to allow compilation
int[] mainCols, // Added return type void to allow compilation
Table refTable, int[] refCols, int type, int deleteAction, int updateAction) {
    core = new ConstraintCore();
    constName = name;
    constType = type;
}