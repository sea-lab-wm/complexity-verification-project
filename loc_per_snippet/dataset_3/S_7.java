    //SNIPPET_STARTS
    public void TestMethodRunner(Object test, Method method, RunNotifier notifier, Description description) {
        super1(test.getClass(), Before.class, After.class, test); // super() renamed to super1() to allow compilation
        fTest= (Ftest) test; // Type cast to Ftest to allow compilation
        fMethod= method;
    } // added to allow compilation

    // Snippet s9
    /**
     * Returns a vector containing the URI (type + path) for all the databases.
     */
