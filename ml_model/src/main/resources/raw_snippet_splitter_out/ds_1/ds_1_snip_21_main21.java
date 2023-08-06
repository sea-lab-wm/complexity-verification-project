package snippet_splitter_out.ds_1;
public class ds_1_snip_21_main21 {
public static void main21(String[] args) {
        int[] array = { 1, 6, 4, 10, 2 };

        for (int i = 0; i <= array.length/2-1; i++){
            int tmp=array[array.length-i-1];
            array[array.length-i-1] = array[i];
            array[i]=tmp;
        }

        for (int i = 0; i <= array.length - 1; i++) { // Note a "{" had to be added here to allow compilation
            System.out.println(array[i]);
        }
    }
}