    //SNIPPET_STARTS    DATASET2START
    public static void main20(String[] args) {
        int i=14;
        String result="";

        while (i>0) {
            if (i%2 ==0)
                result="0"+result;
            else
                result="1"+result;
                i=i/2;
        }

        System.out.println(result); }
    //DATASET2END
    
    // 21. Reverse entries of array                                                 /*Tasks for fMRI-Setting*/
