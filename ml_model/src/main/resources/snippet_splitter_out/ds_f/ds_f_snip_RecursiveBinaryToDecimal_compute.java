package snippet_splitter_out.ds_f;

public class ds_f_snip_RecursiveBinaryToDecimal_compute {
  // SNIPPET_STARTS
  static int compute(String s, int number) {
    if (number < 0) {
      return 0;
    }
    if (s.charAt(number) == '0') {
      return 2 * compute(s, number - 1);
    }
    return 1 + 2 * compute(s, number - 1);
  }
}
