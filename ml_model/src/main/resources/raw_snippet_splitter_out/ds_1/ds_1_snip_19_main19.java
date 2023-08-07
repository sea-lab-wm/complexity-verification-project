package snippet_splitter_out.ds_1;
public class ds_1_snip_19_main19 {
// 19. Capitalize first letter of word                          /*Only in the first pilot study*/
// SNIPPET_STARTS
public static void main19(String[] args) {
    String s = "here are a bunch of words";
    final StringBuilder result = new StringBuilder(s.length());
    String[] words = s.split("\\s");
    for (int i = 0, l = words.length; i < l; ++i) {
        if (i > 0)
            result.append(" ");
        result.append(Character.toUpperCase(words[i].charAt(0))).append(// Note: a ")" had to be added here to allow compilation
        words[i].substring(1));
    }
    System.out.println(result);
}
}