    //SNIPPET_STARTS
    protected Table copyUTable313(UTable uTable)
    {
    Table table = new Table(uTable.getColumnsCount());
    table.setBorderColor(new Color (
    uTable.getBoxModel().getBorderColor().getRed(),
    uTable.getBoxModel().getBorderColor().getGreen(),
    uTable.getBoxModel().getBorderColor().getBlue()));
    Cell cell = null; // Had to be added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    UParagraph paragraph = uCell.getContent();
    if (paragraph.getChildren().size() > 0 &&
    paragraph.getChildren().get(0) instanceof UImage) {
    UImage image = paragraph.getChildren().get(0);
    if (image.getPath().lastSegment().startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (image.getPath().lastSegment().startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(image.getPath().lastSegment());
    }
    }
    if (uCell.getBoxModel().getBackgroundColor() != null) {
    cell.setBackgroundColor(uCell.getBoxModel().getBackgroundColor());
    }
    }
    table.addCell(cell);
    }
    return table;
    }

    // S3_2:1 (Revision 1) Fully resolved method chains, good comments
    /**
    * The images of a unified table are copied to a simplified “yes/no” table.
    */
