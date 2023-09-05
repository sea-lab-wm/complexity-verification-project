// Added to allow compilation
// Snippet s60
// SNIPPET_STARTS
public Object s60() {
    boolean response = warehouseDialog.getResponseBoolean();
    remove(warehouseDialog);
    return response;
}