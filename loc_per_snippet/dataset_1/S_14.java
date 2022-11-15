    public static void main15(String[] args) {
        int array[][] = {{5,6,7},{4,8,9}};
        int array1[][] = {{6,4},{5,7},{1,1}};
        int array2[][] = new int[3][3];

        int x = array.length;
        int y = array1.length;

        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y-1; j++) {
                for(int k = 0; k < y; k++){
                    array2[i][j] += array[i][k]*array1[k][j];
                }
            }
        }

        for(int i = 0; i < x; i++) {
            for(int j = 0; j < y-1; j++) {
                System.out.print(" "+array2[i][j]);
            }
        }
    }
