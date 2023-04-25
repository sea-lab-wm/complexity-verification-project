package FeatureExtraction.snippet_splitter_out;
public class ds_f_snip_lengthOfLastWord_compute {
// SNIPPET_STARTS
public static int compute(String text) {
    int result = 0;
    boolean flag = false;
    for (int i = text.length() - 1; i >= 0; i--) {
        char c = text.charAt(i);
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            flag = true;
            result++;
        } else {
            if (flag)
                break;
        }
    }
    return result;
}
}