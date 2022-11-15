    public void s62() {
        String channelId;

        if (version==1) {
            dataServiceId = (String) in.readObject();
            channelId = "" + in.readInt();
        }
    } // Added to allow compilation
