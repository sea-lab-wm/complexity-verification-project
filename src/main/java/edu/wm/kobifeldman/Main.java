package edu.wm.kobifeldman;

import edu.wm.kobifeldman.fMRI_Study_Classes.*;

public class Main {
    public static void main(String[] args) {
        //@Nullable Object obj = null;  // might be null
        //@NonNull  Object nnobj = new Object();  // never null
        //obj.toString();         // checker warning:  dereference might cause null pointer exception
        //nnobj = obj;           // checker warning:  nnobj may become null
        //if (nnobj == null){ // checker warning:  redundant test

        //}

        //@NonNull Object ref = null;
        //Object ref = null;

        //@Nullable Boolean test = null;

        ArrayAverage.main();
        ContainsSubstring.main();
        YesNo.main();
    }
}
