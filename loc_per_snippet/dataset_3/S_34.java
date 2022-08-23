    //SNIPPET_STARTS
    public class DisbandUnitAction extends Tasks_2{ // Wrapped in a class to allow compilation
        /**
         * Creates a new <code>DisbandUnitAction</code>.
         *
         * @param freeColClient The main controller object for the client.
         */
        DisbandUnitAction(FreeColClient freeColClient) {
            super(freeColClient, "unit.state.8", null, KeyStroke.getKeyStroke('D', 0));
            putValue(BUTTON_IMAGE, freeColClient.getImageLibrary().getUnitButtonImageIcon(ImageLibrary.UNIT_BUTTON_DISBAND,
                    0));
            putValue(BUTTON_ROLLOVER_IMAGE, freeColClient.getImageLibrary().getUnitButtonImageIcon(
                    ImageLibrary.UNIT_BUTTON_DISBAND, 1));
        }
    }

    // Snippet s36
