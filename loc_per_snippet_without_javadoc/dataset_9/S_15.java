    protected Table writeUTable321(UTable uTable)
    {
    /* Create a new table with the same size and color as uTable. */
    Table table = new Table(uTable.getColumnsCount());
    BoxModelOption option = uTable.getBoxModel();
    UColor color = option.getBorderColor();
    table.setBorderColor(new Color
    (color.getRed(),color.getGreen(),color.getBlue()));
    /* Go through all entries of uTable and copy image information. */
    Cell cell = null; // Added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    UParagraph paragraph = uCell.getContent();
    UChildren children = paragraph.getChildren();
    /* Check if there is image information to copy. */
    if (children.size() > 0 && children.get(0) instanceof UImage) {
    UImage image = children.get(0);
    Path path = image.getPath();
    USegment segment = path.lastSegment();
    /* Copy the last segment of the image. False and true */
    /* are transformed to “no” and “yes”, respectively. */
    if (segment.startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (segment.startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(segment);
    }
    }
    /* Copy background color of uCell. */
    option = uCell.getBoxModel();
    color = option.getBackgroundColor();
    if (color != null) {
    cell.setBackgroundColor(color);
    }
    }
    table.addCell(cell);
    }
    return table;
    }
