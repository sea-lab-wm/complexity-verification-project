    //SNIPPET_STARTS
    public Description getDescription() {
        Description spec = Description.createSuiteDescription(getName());
        List<Method> testMethods = fTestMethods;
        for (Method method : testMethods)
            spec.addChild(methodDescription(method));

        return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s64
