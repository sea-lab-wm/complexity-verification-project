    public static void main3(int number1, int number2) {
        int temp; // Note: a ";" had to be added here to allow compilation
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
