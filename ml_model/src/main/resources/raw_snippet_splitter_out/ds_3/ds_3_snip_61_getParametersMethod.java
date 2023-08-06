package snippet_splitter_out.ds_3;
public class ds_3_snip_61_getParametersMethod {
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
    }
}