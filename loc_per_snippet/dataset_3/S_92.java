    //SNIPPET_STARTS
    private Method getParametersMethod() throws Exception {
        for (Method each : fKlass.getMethods()) {
            if (Modifier.isStatic(each.getModifiers())) {
                Annotation[] annotations= each.getAnnotations();
                for (Annotation annotation : annotations) {
                    if (annotation.annotationType().getClass() == Parameters.class) //.getClass() ADDED BY KOBI
                        return each;
                }
            }
        }
        throw new Exception("No public static parameters method on class "
                + getName());
    } // Added to allow compilation

    // Snippet s94
