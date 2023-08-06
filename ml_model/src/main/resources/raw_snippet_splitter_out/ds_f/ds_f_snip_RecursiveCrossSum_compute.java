package snippet_splitter_out.ds_f;
public class ds_f_snip_RecursiveCrossSum_compute {
public static int compute(int number) {
        if (number == 0) {
            return 0;
        }

        return (number % 10) + compute((int) number/10);
    }
}