// 6. Sum from 1 to n                                                          /*First and second
// pilot study*/
// SNIPPET_STARTS
public static void main6(String[] args) {
    // Note: a ";" had to be added here to allow compilation
    int n = 4;
    int result = 0;
    for (int i = 1; i <= n; i++) result = result + i;
    System.out.println(result);
}