package snippet_splitter_out.ds_1;
public class ds_1_snip_2_main2 {
public static void main2(String[] args) {
        String string1 = "Magdeburg";
        String string2 = "Hamburg";

        int length;
        if (string1.length() < string2.length())
            length = string1.length();
        else length = string2.length();
        
        int counter=0;
        
        for (int i = 0; i < length; i++) {
            if (string1.charAt(i) == string2.charAt(i)) {
                counter++;
            }
        }
        System.out.println(counter);
    }
}