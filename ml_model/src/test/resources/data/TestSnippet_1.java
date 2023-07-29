package data;
public class TestSnippet_1 {

  /**
    * this is a javadoc comment! #1
    * use this test snippet to create test cases
    */
  public void TestLoops(int x, int y) { //I think it's funny that x and y aren't actually used in the snippet. #2
    for (int i = 0; i < 10; i++) {
      System.out.println("single_for");
    }

    int i = 0;
    while (i < 10) {
      System.out.println("single_while");
      i++;
    }

    /*
    This is a block comment!
    #3
    */    
    for (int j = 0; j < 10; j++) {
      for (int k = 0; k < 10; k++) {
        System.out.println("nested_for");
      }
    }

    //This is also treated like a block comment, but a little different!
    //In reality, the extra lines are orphan comments and are not counted towards the total.
    //#4
    int k = 0, j = 0;
    while (k < 10) {
      while (j < 10) {
        System.out.println("nested_while");
      }
    }
    //This one looks a little complicated. I'm not analyzing it. #5
    int i1 = 0;
    while (i1 < 10) {
      for (int i2 = 0; i2 < 10l; i2++) {
        System.out.println("nested_while_for");
        for (i1 = 1; i1 < 5; i1++) {
          System.out.println("nested_while_for_depth");
        }
      }
    }

    if (true) { 
      System.out.println("single_if"); //Looks like this will always execute! #6
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
        System.out.println("nested_if"); //This is dead code and will never execute. #7
      } else {
        System.out.println("else_if");
      }
    }
  }
}
