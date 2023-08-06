package snippet_splitter_out.ds_2;
public class ds_2_snip_4_main8 {
public static void main8(String[] args) {
        int number = 323;
        int result = 0;

        while (number!= 0) {
            result = result + number % 10;
            number = number / 10;
        }
        System.out.println(result);
    }
}