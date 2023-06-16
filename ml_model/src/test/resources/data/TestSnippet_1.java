
package data;

public class TestSnippet_1 {

    public void TestLoops(int x, int y){
        for (int i=0; i< 10; i++){
            System.out.println("single_for");
        }

        int i = 0;
        while (i < 10){
            System.out.println("single_while");
            i++;
        }

        for (int j=0; j< 10; j++){
            for (int k=0; k<10; k++){
                System.out.println("nested_for");
            }
        }

        int k = 0, j = 0;
        while (k<10){
            while (j<10){
                System.out.println("nested_while");
            }
        }

        int i1 = 0;
        while (i1<10){
            for (int i2=0 ; i2<10l; i2++){
                System.out.println("nested_while_for");
                for (i1 = 1; i1<5 ;i1++) {
                    System.out.println("nested_while_for_depth");
                }
            }
        }

        if (true) {
            System.out.println("single_if");
        }

        if (true) {
            if (true) {
                System.out.println("nested_if");
            }
        }

        if (true) {
            if (true) {
                System.out.println("nested_if");
            } else if (true) {
                System.out.println("nested_if");
            } else {
                System.out.println("else_if");
            }
        }
    }
}