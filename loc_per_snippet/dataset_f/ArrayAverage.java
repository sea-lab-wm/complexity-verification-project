    //SNIPPET_STARTS
    public static float compute(int[] numbers) {
        int number1 = 0;
        int number2 = 0;

        while (number1 < numbers.length) {
            number2 = number2 + numbers[number1];
            number1 = number1 + 1;
        }

        float result = number2 / (float) number1;
        return result;
    }
}