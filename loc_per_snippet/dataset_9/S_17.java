    protected Table writeUTable323(UTable uTable)
    {
    Table table = new Table(uTable.getColumnsCount());
    BoxModelOption option = uTable.getBoxModel();
    UColor color = option.getBorderColor();
    table.setBorderColor(new Color
    (color.getRed(),color.getGreen(),color.getBlue()));
    Cell cell = null; // Added to allow compilation
    for (UTableCell uCell : uTable.getEntries()) {
    UParagraph paragraph = uCell.getContent();
    UChildren children = paragraph.getChildren();
    if (children.size() > 0 && children.get(0) instanceof UImage) {
    UImage image = children.get(0);
    Path path = image.getPath();
    USegment segment = path.lastSegment();
    if (segment.startsWith("false")) {
    cell = new Cell("no");
    } else {
    if (segment.startsWith("true")) {
    cell = new Cell("yes");
    } else {
    cell = new Cell(segment);
    }
    }
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
