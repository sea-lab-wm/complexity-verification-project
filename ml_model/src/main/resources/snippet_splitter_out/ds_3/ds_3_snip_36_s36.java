package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_36_s36 {
// Snippet s36
// SNIPPET_STARTS
public Object s36() {
    Class clas = object.getClass();
    Field field = Reflect.resolveJavaField(clas, name, false);
    if (field != null)
        return new Variable(name, field.getType(), new LHS(object, field));
    return object;
    /*Altered return*/
    // return null; // Added to allow compilation
}
}