    //SNIPPET_STARTS
    public void s52() {

        panel.add(UiUtilities.createHelpTextArea(mLocalizer.msg("help","No endtime defined")), cc.xy(1,1));

        mTimePanel = new TimeDateChooserPanel(date);
        panel.add(mTimePanel, cc.xy(1,3));

    } // Added to allow compilation

    // Snippet s53
