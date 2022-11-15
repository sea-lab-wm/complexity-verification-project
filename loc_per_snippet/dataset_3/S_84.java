    /**
     * Adds a message to the list of messages that need to be displayed on the GUI.
     * @param message The message to add.
     */
    public synchronized void addMessage(GUIMessage message) {
        if (getMessageCount() == MESSAGE_COUNT) {
            messages2.remove(0); // Renamed to allow compilation
        }
        messages2.add(message); // Renamed to allow compilation

        freeColClient.getCanvas().repaint(0, 0, getWidth(), getHeight());
    } // Added to allow compilation
