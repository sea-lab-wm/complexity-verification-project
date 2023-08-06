package snippet_splitter_out.ds_f;
public class ds_f_snip_isPalindrome_compute {
public static boolean compute(String word) {
        boolean result = true;

        for (int i = 0, j = word.length() - 1; i < word.length() / 2; i++, j--) {
            if (word.charAt(i) != word.charAt(j)) {
                result = false;
                break;
            }
        }

        return result;
    }
}