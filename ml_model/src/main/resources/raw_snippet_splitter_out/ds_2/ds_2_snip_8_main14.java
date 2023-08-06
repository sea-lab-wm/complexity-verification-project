package snippet_splitter_out.ds_2;
public class ds_2_snip_8_main14 {
public static void main14(String[] args) {
        String word = "Hello";
        String result = new String();

        for ( int j = word.length() - 1; j >= 0; j-- )
            result += word.charAt(j);
        
        System.out.println(word);
    }
}