    //SNIPPET_STARTS
    public This getGlobal( Interpreter declaringInterpreter )
    {
        if ( parent != null )
            return parent.getGlobal( declaringInterpreter );
        else
            return getThis( declaringInterpreter );
    } // Added to allow compilation

    // Snippet s47
    /**
     * Read Settings
     * @param stream
     * @throws IOException
     * @throws ClassNotFoundException
     */
