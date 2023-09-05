// added to allow compilation
// Snippet s8
// SNIPPET_STARTS
public void TestMethodRunner(Object test, Method method, RunNotifier notifier, Description description) {
    // super() renamed to super1() to allow compilation
    super1(test.getClass(), Before.class, After.class, test);
    // Type cast to Ftest to allow compilation
    fTest = (Ftest) test;
    fMethod = method;
}