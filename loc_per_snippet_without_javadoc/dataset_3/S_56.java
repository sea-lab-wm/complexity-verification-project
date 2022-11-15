    public void s57() {
        String classname = (String) in.readObject();
        String devname = (String)in.readObject();

        DeviceIf dev = DriverFactory.getInstance().createDevice(classname, devname);
    } // Added to allow compilation
