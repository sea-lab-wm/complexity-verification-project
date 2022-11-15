    public ElgatoChannel[] getAvailableChannels() {
        ArrayList<ElgatoChannel> list = new ArrayList<ElgatoChannel>();

        String res = null;
        try {
            res = mAppleScript.executeScript(CHANNELLIST);
        } finally {
            // Added to allow compilation
        }
        return new ElgatoChannel[0];                                                /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation
