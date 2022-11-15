    public static Vector getDatabaseURIs() {

        Vector v = new Vector();
        Iterator it = databaseIDMap.values().iterator();

        while (it.hasNext()) {
            Database db = (Database) it.next();
        } // added to allow compilation
        return v;                                                       /*Altered return*/
        //return null; // added to allow compilation
    } // added to allow compilation
