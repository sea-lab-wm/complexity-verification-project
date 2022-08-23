    //SNIPPET_STARTS    DATASET2START
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
    //DATASET2END

    // 22. Median on sorted data                                                /*Tasks for fMRI-Setting*/
