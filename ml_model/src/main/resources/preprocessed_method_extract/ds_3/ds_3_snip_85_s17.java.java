// Snippet s17
// SNIPPET_STARTS
public void s17() {
    mDevices = new Vector<DeviceIf>();
    DeviceFileHandling reader = new DeviceFileHandling();
    for (int i = 0; i < num; i++) {
        String classname = (String) in.readObject();
    }
}