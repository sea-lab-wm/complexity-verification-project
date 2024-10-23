package snippet_splitter_out.ds_f;

public class ds_f_snip_CountVowels_compute {
  // SNIPPET_STARTS
  public static int compute(String word) {
    char[] letters = {'a', 'e', 'i', 'o', 'u'};
    int result = 0;
    for (int i = 0; i < word.length(); i++) {
      for (int j = 0; j < letters.length; j++) {
        if (word.charAt(i) == letters[j]) {
          result++;
        }
      }
    }
    return result;
  }
}
