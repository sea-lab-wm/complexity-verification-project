    public String compact(String message) {
        if (fExpected == null || fActual == null || areStringsEqual())
            return Assert.format(message, fExpected, fActual);

        findCommonPrefix();
        findCommonSuffix();

        return message;                                                                 /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation
