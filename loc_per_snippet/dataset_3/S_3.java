    /**
     * Constructor, with a argument reference to the PUBLIC User Object which
     * is null if this is the SYS or PUBLIC user.
     *
     * The dependency upon a GranteeManager is undesirable.  Hopefully we
     * can get rid of this dependency with an IOC or Listener re-design.
     */
    public void Grantee(String name, Grantee inGrantee, // public void added to allow compilation
            GranteeManager man) throws HsqlException {

        rightsMap = new IntValueHashMap();
        granteeName = name;
        granteeManager = man;
    }
