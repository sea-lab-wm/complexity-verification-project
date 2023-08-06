package snippet_splitter_out.ds_3;
public class ds_3_snip_71_compactString {
private String compactString(String source) {
        String result = DELTA_START + source.substring(fPrefix, source.length() - fSuffix + 1) + DELTA_END;
        if (fPrefix > 0)
            result = computeCommonPrefix() + result;
        if (fSuffix > 0)
            result = result + computeCommonSuffix();
        return result; // had to be added to allow compilation
    }
}