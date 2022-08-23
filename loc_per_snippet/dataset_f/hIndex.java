    //SNIPPET_STARTS
    public static int compute(int[] numbers) {
        Arrays.sort(numbers);

        int count = numbers.length;
        int result = 0;
        for (int i = count - 1; i >= 0; i--) {
            int current = count - i;
            if (numbers[i] >= current) {
                result = current;
            } else {
                break;
            }
        }

        return result;
    }
}