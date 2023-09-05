// 5. Binary search                                     /*Only in the first pilot study*/
// SNIPPET_STARTS
public static void main5(String[] args) {
    int[] array = { 2, 4, 5, 6, 8, 10, 13 };
    int key = 5;
    int index1 = 0;
    int index2 = array.length - 1;
    while (index1 <= index2) {
        int m = (index1 + index2) / 2;
        if (key < array[m])
            index2 = m - 1;
        else if (key > array[m])
            index1 = m + 1;
        else {
            System.out.println(m);
            break;
        }
    }
}