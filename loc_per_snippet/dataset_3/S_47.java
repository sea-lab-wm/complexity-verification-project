    //SNIPPET_STARTS
    public class ComparisonFailure extends Tasks_2{ // Class wrapper to allow compilation
        /**
         * Constructs a comparison failure.
         * @param message the identifying message or null
         * @param expected the expected string value
         * @param actual the actual string value
         */
        public ComparisonFailure (String message, String expected, String actual) {
            super (message);
            fExpected= expected;
            fActual= actual;
        } // Added to allow compilation
    }

    // Snippet s49
