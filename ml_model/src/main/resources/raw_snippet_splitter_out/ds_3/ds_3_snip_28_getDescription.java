package snippet_splitter_out.ds_3;
public class ds_3_snip_28_getDescription {
public Description getDescription() {
        Description spec = Description.createSuiteDescription(getName());
        List<Method> testMethods = fTestMethods;
        for (Method method : testMethods)
            spec.addChild(methodDescription(method));

        return null; // Added to allow compilation
    }
}