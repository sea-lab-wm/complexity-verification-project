package snippet_splitter_out.ds_2;

public class ds_2_snip_10_main20 {
  // 20. Decimal to binary                                                                /*Tasks
  // for fMRI-Setting*/
  // SNIPPET_STARTS    DATASET2START
  public static void main20(String[] args) {
    int i = 14;
    String result = "";
    while (i > 0) {
      if (i % 2 == 0) result = "0" + result;
      else result = "1" + result;
      i = i / 2;
    }
    System.out.println(result);
  }
}
