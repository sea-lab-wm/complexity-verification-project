    public void s76() throws IOException{
        out.writeObject(device.getDriver().getClass().getName());
        out.writeObject(device.getName());

        device.writeData(out);
    } // Added to allow compilation
