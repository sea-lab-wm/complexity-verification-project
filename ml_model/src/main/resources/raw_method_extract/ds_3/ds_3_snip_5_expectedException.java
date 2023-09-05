// Snippet s38                                                                      /*ORIGINALLY COMMENTED OUT, ALTERED METHOD*/
// SNIPPET_STARTS
Class<? extends Throwable> expectedException(Method method) {
    Test annotation = method.getAnnotation(Test.class);
    if (annotation.expected() == None.class)
        return null;
    else
        return annotation.expected();
}