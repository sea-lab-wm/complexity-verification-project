package snippet_splitter_out.ds_2;

public class ds_2_snip_3_main7 {
  // 7. Find max in list of numbers                                           /*Tasks for
  // fMRI-Setting*/
  // SNIPPET_STARTS    DATASET2START
  public static void main7(String[] args) {
    int[] array = {2, 19, 5, 17};
    int result = array[0];
    for (int i = 1; i < array.length; i++) if (array[i] > result) result = array[i];
    System.out.println(result);
  }
}
