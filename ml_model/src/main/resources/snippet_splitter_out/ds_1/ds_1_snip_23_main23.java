package snippet_splitter_out.ds_1;

public class ds_1_snip_23_main23 {
  // DATASET2END
  // 23. Double entries of array                              /*First and second pilot study*/
  // SNIPPET_STARTS
  public static void main23(String[] args) {
    int[] array = {1, 3, 11, 7, 4};
    for (int i = 0; i < array.length; i++) array[i] = array[i] * 2;
    for (int i = 0; i <= array.length - 1; i++) System.out.println(array[i]);
  }
}
