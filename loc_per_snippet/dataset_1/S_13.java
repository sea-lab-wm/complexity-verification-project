    //SNIPPET_STARTS    DATASET2START
    public static void main14(String[] args) {
        String word = "Hello";
        String result = new String();

        for ( int j = word.length() - 1; j >= 0; j-- )
            result += word.charAt(j);
        
        System.out.println(word);
    }
    //DATASET2END

    // 15. Matrix multiplication                                    /*Only in the first pilot study*/
