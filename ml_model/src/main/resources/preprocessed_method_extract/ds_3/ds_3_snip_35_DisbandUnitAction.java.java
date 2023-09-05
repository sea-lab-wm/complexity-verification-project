// SNIPPET_STARTS
DisbandUnitAction(FreeColClient freeColClient) {
    super(freeColClient, "unit.state.8", null, KeyStroke.getKeyStroke('D', 0));
    putValue(BUTTON_IMAGE, freeColClient.getImageLibrary().getUnitButtonImageIcon(ImageLibrary.UNIT_BUTTON_DISBAND, 0));
    putValue(BUTTON_ROLLOVER_IMAGE, freeColClient.getImageLibrary().getUnitButtonImageIcon(ImageLibrary.UNIT_BUTTON_DISBAND, 1));
}