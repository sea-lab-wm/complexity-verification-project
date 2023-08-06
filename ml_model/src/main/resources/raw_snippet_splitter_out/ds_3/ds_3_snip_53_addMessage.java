package snippet_splitter_out.ds_3;
public class ds_3_snip_53_addMessage {
public synchronized void addMessage(GUIMessage message) {
        if (getMessageCount() == MESSAGE_COUNT) {
            messages2.remove(0); // Renamed to allow compilation
        }
        messages2.add(message); // Renamed to allow compilation

        freeColClient.getCanvas().repaint(0, 0, getWidth(), getHeight());
    }
}