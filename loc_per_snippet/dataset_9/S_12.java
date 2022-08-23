    //SNIPPET_STARTS
    protected Table copyUTable311(UTable uTable)
    {
    /* Create a new table with the same size and color as uTable. */
    Table table = new Table(uTable.getColumnsCount());
    table.setBorderColor(new Color (
    uTable.getBoxModel().getBorderColor().getRed(),
    uTable.getBoxModel().getBorderColor().getGreen(),
    uTable.getBoxModel().getBorderColor().getBlue()));
    /* Go through all entries of uTable and copy image information. */
    Cell cell = null; // had to be added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    UParagraph paragraph = uCell.getContent();
    /* Check if there is image information to copy. */
    if (paragraph.getChildren().size() > 0 &&
    paragraph.getChildren().get(0) instanceof UImage) {
    UImage image = paragraph.getChildren().get(0);
    /* Copy the last segment of the image. False and true */
    /* are transformed to “no” and “yes”, respectively. */
    if (image.getPath().lastSegment().startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (image.getPath().lastSegment().startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(image.getPath().lastSegment());
    }
    }
    /* Copy background color of uCell. */
    if (uCell.getBoxModel().getBackgroundColor() != null) {
    cell.setBackgroundColor(uCell.getBoxModel().getBackgroundColor());
    }
    }
    table.addCell(cell);
    }
    return table;
    }

    // S3_1:2 (Revision 1) method chains, bad comments
    /**
    * The images of a unified table are copied to a simplified “yes/no” table.
    */
