    //SNIPPET_STARTS
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

    // Snippet 4
    // raptor.chess.pgn.PgnUtils.getMove
    // http://code.google.com/p/raptor‐chess‐interface/source/browse/tags/v.92/src/raptor/chess/pgn/PgnUtils.jav
    // a

    // S4_1:1 method chains, good comments
    /**
    * Generates a string (“builder”) for a given chess move in PGN (Portable
    * Game Notation). This includes the move number and all NAG annotations.
    */
