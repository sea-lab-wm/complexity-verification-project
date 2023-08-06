package snippet_splitter_out.ds_1;
public class ds_1_snip_18_main18 {
public static void main18(String[] args) {
        int number1 = 23;
        int number2 = 42;

        int max, min;
        int results = -1; // Note: a ";" had to be added here to allow compilation

        if (number1>number2) {
            max = number1; min = number2;
        } else {
            max = number2; min = number1;
        }
        for(int i=1; i<=min; i++) {
            if( (max*i)%min == 0 ) {
                results = i*max; break; // Note: result had to be renamed to results to allow compilation
            }
        }
        if(results != -1) // Note: result had to be renamed to results to allow compilation
            System.out.println(results);
        else
            System.out.println("Error!");
    }
}