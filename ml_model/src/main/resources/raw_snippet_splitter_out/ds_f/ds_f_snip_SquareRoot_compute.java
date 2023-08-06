package snippet_splitter_out.ds_f;
public class ds_f_snip_SquareRoot_compute {
public static String compute(int[] numbers) {
        double[] result = new double[numbers.length];

        for (int i = 0; i < numbers.length; i++) {
            if (numbers[i] == 0) {
                result[i] = 0;
                continue;
            }

            if (numbers[i] < 0) {
                result[i] = Math.sqrt(-1 * numbers[i]);
            } else {
                result[i] = Math.sqrt(numbers[i]);
            }
        }

        return Arrays.toString(result);
    }
}