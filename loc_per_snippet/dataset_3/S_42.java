    //SNIPPET_STARTS
    public class TestClassRunnerForParameters extends Tasks_2{ // Added class wrapper to allow compilation
        private TestClassRunnerForParameters(Class<?> klass, Object[] parameters, int i) {
            super(klass);
            fParameters= parameters;
            fParameterSetNumber= i;
        } // Added to allow compilation
    }

    // Snippet s44
