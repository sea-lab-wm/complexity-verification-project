// DATASET2END
// 3. Greatest common divisor                                               /*Only in the first pilot study*/
// Note: To allow compilation, number1 and number2 had to be defined
// SNIPPET_STARTS
public static void main3(int number1, int number2) {
    // Note: a ";" had to be added here to allow compilation
    int temp;
    do {
        if (number1 < number2) {
            temp = number1;
            number1 = number2;
            number2 = temp;
        }
        temp = number1 % number2;
        if (temp != 0) {
            number1 = number2;
            number2 = temp;
        }
    } while (temp != 0);
    System.out.println(number2);
}