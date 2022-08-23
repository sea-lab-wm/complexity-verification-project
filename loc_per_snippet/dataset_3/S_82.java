    //SNIPPET_STARTS
    public void filter2(Filter filter) throws NoTestsRemainException { // Renamed to allow compilation
        for (Iterator<Method> iter= fTestMethods.iterator(); iter.hasNext();) {
            Method method= iter.next();
            if (!filter.shouldRun(methodDescription(method)))
                iter.remove();
        }
        if (fTestMethods.isEmpty())
            throw new NoTestsRemainException();
    } // Added to allow compilation

    // Snippet s84
