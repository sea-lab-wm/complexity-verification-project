package snippet_splitter_out.ds_1;

public class ds_1_snip_14_main14 {
  // DATASET2END
  // 14. Reverse string                                                       /*Tasks for
  // fMRI-Setting*/
  // SNIPPET_STARTS    DATASET2START
  public static void main14(String[] args) {
    String word = "Hello";
    String result = new String();
    for (int j = word.length() - 1; j >= 0; j--) result += word.charAt(j);
    System.out.println(word);
  }
}
