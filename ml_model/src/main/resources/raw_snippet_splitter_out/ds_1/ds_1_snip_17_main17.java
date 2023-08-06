package snippet_splitter_out.ds_1;
public class ds_1_snip_17_main17 {
public static void main17(String[] args) {
        String word = "Programming in Java";
        String key1 = "Java";
        String key2 = "Pascal";

        int index1 = word.indexOf(key1);
        int index2 = word.indexOf(key2);

        if (index1 != -1)
            System.out.println("Substring is contained: " + key1);
        else
            System.out.println("Substring is not contained: " + key1);
        if (index2 != -1)
            System.out.println("Substring is contained: " + key2);
        else
            System.out.println("Substring is not contained: " + key2);
    }
}