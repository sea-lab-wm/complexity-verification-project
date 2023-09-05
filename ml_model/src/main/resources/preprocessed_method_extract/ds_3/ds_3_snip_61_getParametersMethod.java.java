// Added to allow compilation
// Snippet s93                                                                  /*ORIGINALLY
// COMMENTED OUT*/
// SNIPPET_STARTS
private Method getParametersMethod() throws Exception {
    for (Method each : fKlass.getMethods()) {
        if (Modifier.isStatic(each.getModifiers())) {
            Annotation[] annotations = each.getAnnotations();
            for (Annotation annotation : annotations) {
                if (// .getClass() ADDED BY KOBI
                annotation.annotationType().getClass() == Parameters.class)
                    return each;
            }
        }
    }
    throw new Exception("No public static parameters method on class " + getName());
}