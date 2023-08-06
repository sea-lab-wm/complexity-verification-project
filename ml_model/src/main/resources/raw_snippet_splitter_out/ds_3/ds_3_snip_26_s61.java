package snippet_splitter_out.ds_3;
public class ds_3_snip_26_s61 {
public void s61() {
		if (true) // Added to allow compilation
            System.out.println(""); // Added to allow compilation
        else if ( returnType.equals("F") )
            opcode = FRETURN;
        else if ( returnType.equals("J") )  //long
            opcode = LRETURN;

        cv.visitInsn(opcode);
    }
}