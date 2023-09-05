// Added to allow compilation
// Snippet s76
// SNIPPET_STARTS
public void s76() throws IOException {
    out.writeObject(device.getDriver().getClass().getName());
    out.writeObject(device.getName());
    device.writeData(out);
}