package snippet_splitter_out.ds_1;
public class ds_1_snip_12_main12 {
public static void main12(String[] args) {
        String word = "otto";
        boolean result = true;
        for (int i = 0, j = word.length() - 1; i < word.length()/2; i++, // Note: "int" before j had to be removed to allow compilation
            j--) {
            if (word.charAt(i) != word.charAt(j)) {
                result = false;
                break;
            }
        }
        System.out.println(result);
    }
}