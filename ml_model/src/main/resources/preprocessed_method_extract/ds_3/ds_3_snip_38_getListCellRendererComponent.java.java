// SNIPPET_STARTS
public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
    JLabel label = (JLabel) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);
    String str;
    if (value instanceof DeviceIf) {
        DeviceIf device = (DeviceIf) value;
    }
    // Added to allow compilation
    return new Component(str = "");
    /*Altered return*/
    // return null; // Added to allow compilation
}