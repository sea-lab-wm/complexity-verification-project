// Added to allow compilation
// Snippet s61
// SNIPPET_STARTS
public void s61() {
    if (// Added to allow compilation
    true)
        // Added to allow compilation
        System.out.println("");
    else if (returnType.equals("F"))
        opcode = FRETURN;
    else if (// long
    returnType.equals("J"))
        opcode = LRETURN;
    cv.visitInsn(opcode);
}