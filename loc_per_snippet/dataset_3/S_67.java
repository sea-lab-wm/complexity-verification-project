    //SNIPPET_STARTS
    public void run(RunNotifier notifier) {
        TestResult result= new TestResult();
        result.addListener(createAdaptingListener(notifier));
        fTest.run(result);
    } // Added to allow compilation


    // Snippet s69
