    public void Grantee(String name, Grantee inGrantee, // public void added to allow compilation
            GranteeManager man) throws HsqlException {

        rightsMap = new IntValueHashMap();
        granteeName = name;
        granteeManager = man;
    }
