    //SNIPPET_STARTS
    public NameSpace swap( NameSpace newTop ) {
        NameSpace oldTop = (NameSpace)(stack.elementAt(0));
        stack.setElementAt( newTop, 0 );
        return oldTop;
    } // Added to allow compilation

    // Snippet s57
