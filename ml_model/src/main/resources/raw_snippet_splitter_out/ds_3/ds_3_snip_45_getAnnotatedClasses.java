package snippet_splitter_out.ds_3;
public class ds_3_snip_45_getAnnotatedClasses {
private static Class<?>[] getAnnotatedClasses(Class<?> klass) throws InitializationError {
            SuiteClasses annotation= klass.getAnnotation(SuiteClasses.class);
            if (annotation == null)
                throw new Tasks_3("message").new InitializationError(String.format("class '%s' must have a SuiteClasses annotation", klass.getName()));
            return annotation.value();
    }
}