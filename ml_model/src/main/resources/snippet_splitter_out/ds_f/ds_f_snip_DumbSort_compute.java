package snippet_splitter_out.ds_f;

public class ds_f_snip_DumbSort_compute {
  // SNIPPET_STARTS
  public static List<Integer> compute(int a, int b, int c, int d) {
    if (a > b) {
      int temp = b;
      b = a;
      a = temp;
    }
    if (c > d) {
      int temp = d;
      d = c;
      c = temp;
    }
    if (a > c) {
      int temp = c;
      c = a;
      a = temp;
    }
    if (b > d) {
      int temp = d;
      d = b;
      b = temp;
    }
    if (b > c) {
      int temp = c;
      c = b;
      b = temp;
    }
    return Arrays.asList(a, b, c, d);
  }
}
