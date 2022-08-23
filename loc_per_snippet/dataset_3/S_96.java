    //SNIPPET_STARTS
    public ActionMenu getButtonAction() {
        AbstractAction action = new AbstractAction() {

            public void actionPerformed(ActionEvent evt) {
                showDialog();
            }
        };
        action.putValue(Action.NAME, mLocalizer.msg("CapturePlugin", "Capture Plugin"));
        action.putValue(Action.SMALL_ICON, createImageIcon("mimetypes", "video-x-generic", 16));

        return new ActionMenu();                                                                                                    /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s98
