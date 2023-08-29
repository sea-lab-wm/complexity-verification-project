package snippet_splitter_out.ds_1;

public class ds_1_snip_4_main4 {
  // 4. BubbleSort                                                        /*Only in the first pilot
  // study*/
  // SNIPPET_STARTS
  public static void main4(String[] args) {
    int[] array = {14, 5, 7};
    for (int counter1 = 0; counter1 < array.length; counter1++) {
      for (int counter2 = counter1; counter2 > 0; counter2--) {
        if (array[counter2 - 1] > array[counter2]) {
          int variable1 = array[counter2];
          array[counter2] = array[counter2 - 1];
          array[counter2 - 1] = variable1;
        }
      }
    }
    for (int counter3 = 0; counter3 < array.length; counter3++) System.out.println(array[counter3]);
  }
}
