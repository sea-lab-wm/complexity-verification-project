    //SNIPPET_STARTS    DATASET2START
    public static void main22(String[] args) {
        int[] array={1,2,4,5,6,10};

        Arrays.sort(array); // Note: this line had to me changed to allow compilation

        float b;
        if (array.length % 2==1)
            b=array[array.length /2];
        else
            b=(array[array.length/2-1]+array[array.length/2])/2f;
        
        System.out.println(b);
    }
