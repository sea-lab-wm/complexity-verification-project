// Snippet s22
/**
 * Returns the PluginPanel
 *
 * @return Panel
 */
// SNIPPET_STARTS
public JPanel createSettingsPanel() {
    mPanel = new CapturePluginPanel(mOwner, mCloneData);
    mPanel.setBorder(Borders.createEmptyBorder(Sizes.DLUY5, Sizes.DLUX5, Sizes.DLUY5, Sizes.DLUX5));
    mPanel.setSelectedTab(mCurrentPanel);
    return new JPanel();
    /*Altered return*/
    // return null; // Added to allow compilation
}