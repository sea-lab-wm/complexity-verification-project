package snippet_splitter_out.ds_1;

public class ds_1_snip_12_main12 {
  // DATASET2END
  // 12. Check palindrom                                          /*First and second pilot study*/
  // SNIPPET_STARTS
  public static void main12(String[] args) {
    String word = "otto";
    boolean result = true;
    for ( // Note: "int" before j had to be removed to allow compilation
    int i = 0, j = word.length() - 1; // Note: "int" before j had to be removed to allow compilation
        i < word.length() / 2; // Note: "int" before j had to be removed to allow compilation
        i++, j--) {
      if (word.charAt(i) != word.charAt(j)) {
        result = false;
        break;
      }
    }
    System.out.println(result);
  }
}
