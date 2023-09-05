// Snippet s4
/**
 * Constructor, with a argument reference to the PUBLIC User Object which is null if this is the
 * SYS or PUBLIC user.
 *
 * <p>The dependency upon a GranteeManager is undesirable. Hopefully we can get rid of this
 * dependency with an IOC or Listener re-design.
 */
// SNIPPET_STARTS
public // public void added to allow compilation
// public void added to allow compilation
void // public void added to allow compilation
Grantee(// public void added to allow compilation
String name, Grantee inGrantee, GranteeManager man) throws HsqlException {
    rightsMap = new IntValueHashMap();
    granteeName = name;
    granteeManager = man;
}