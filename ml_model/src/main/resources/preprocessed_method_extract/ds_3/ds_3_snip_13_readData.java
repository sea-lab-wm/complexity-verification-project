// Added to allow compilation
// Snippet s47
/**
 * Read Settings
 *
 * @param stream
 * @throws IOException
 * @throws ClassNotFoundException
 */
// SNIPPET_STARTS
public void readData(ObjectInputStream stream) throws IOException, ClassNotFoundException {
    int version = stream.readInt();
    mNumber = stream.readInt();
    mName = stream.readUTF();
}