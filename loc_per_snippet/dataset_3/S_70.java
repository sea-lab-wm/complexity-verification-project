    //SNIPPET_STARTS
    static int getASMModifiers( Modifiers modifiers )
    {
        int mods = 0;
        if ( modifiers == null )
            return mods;

        if ( modifiers.hasModifier("public") )
            mods += ACC_PUBLIC;
        return 0; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s72
