    public Object s60() {
        boolean response = warehouseDialog.getResponseBoolean();
        remove(warehouseDialog);
        return response;
    } // Added to allow compilation
