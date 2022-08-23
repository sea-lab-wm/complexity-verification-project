    //SNIPPET_STARTS
    public void readData(ObjectInputStream stream) throws IOException, ClassNotFoundException {
        int version = stream.readInt();
        mNumber = stream.readInt();
        mName = stream.readUTF();
    } // Added to allow compilation

    // Snippet s48
