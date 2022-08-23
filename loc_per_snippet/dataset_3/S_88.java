    //SNIPPET_STARTS
    String getStateString() {

        int state = getState();

        switch (state) {

            case DATABASE_CLOSING:
                return "DATABASE_CLOSING";

            case DATABASE_ONLINE:
                return "DATABASE_ONLINE";
        } // Added to allow compilation
        return new String();                                                           /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s90
