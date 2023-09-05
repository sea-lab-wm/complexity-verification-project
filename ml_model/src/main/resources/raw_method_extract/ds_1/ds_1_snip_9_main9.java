// DATASET2END
// 9. Prime test                                                       /*Tasks for fMRI-Setting*/
// SNIPPET_STARTS
public static void main9(String[] args) {
    int number = 11;
    boolean result = true;
    for (int i = 2; i < number; i++) {
        if (number % i == 0) {
            result = false;
            break;
        }
    }
    System.out.println(result);
}