    //SNIPPET_STARTS
    public boolean isReadOnly() throws HsqlException {

        Object info = getAttribute(Session.INFO_CONNECTION_READONLY);

        isReadOnly = ((Boolean) info).booleanValue();

        return isReadOnly;
    } // Added to allow compilation

    // Snippet s60
