    //SNIPPET_STARTS
    public void s55() {
        if (missionChip == null) {
            GraphicsConfiguration gc = GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice()
                    .getDefaultConfiguration();
            loadMissionChip(gc, color, expertMission);

            if (expertMission) {
                missionChip = expertMissionChips.get(color);
            } // Added to allow compilation
        } // Added to allow compilation
    } // Added to allow compilation


    // Snippet s56
    /**
     Swap in the value as the new top of the stack and return the old
     value.
     */
