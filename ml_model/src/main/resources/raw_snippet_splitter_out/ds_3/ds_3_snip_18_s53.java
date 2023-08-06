package snippet_splitter_out.ds_3;
public class ds_3_snip_18_s53 {
public void s53() throws Exception{
        try {
            suiteMethod= klass.getMethod("suite");
            if (! Modifier.isStatic(suiteMethod.getModifiers())) {
                throw new Exception(klass.getName() + ".suite() must be static");
            }
            suite= (Test) suiteMethod.invoke(null); // static method

        } finally {
            // Added to allow compilation
        }
    }
}