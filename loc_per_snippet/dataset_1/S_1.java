    //SNIPPET_STARTS    DATASET2START
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
    //DATASET2END

    // 3. Greatest common divisor                                               /*Only in the first pilot study*/
    // Note: To allow compilation, number1 and number2 had to be defined
