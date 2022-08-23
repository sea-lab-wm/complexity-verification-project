    //SNIPPET_STARTS
    public void s61() {
		if (true) // Added to allow compilation
            System.out.println(""); // Added to allow compilation
        else if ( returnType.equals("F") )
            opcode = FRETURN;
        else if ( returnType.equals("J") )  //long
            opcode = LRETURN;

        cv.visitInsn(opcode);
    } // Added to allow compilation

    // Snippet s62
