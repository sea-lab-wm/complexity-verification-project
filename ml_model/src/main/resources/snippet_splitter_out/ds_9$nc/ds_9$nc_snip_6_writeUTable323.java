package snippet_splitter_out.ds_9$nc;
public class ds_9$nc_snip_6_writeUTable323 {
// SNIPPET_END_2
// S3_2:3 (Revision 1) Fully resolved method chains, no comments
/**
 * The images of a unified table are copied to a simplified “yes/no” table.
 */
// SNIPPET_STARTS_3
protected Table writeUTable323(UTable uTable) {
    Table table = new Table(uTable.getColumnsCount());
    BoxModelOption option = uTable.getBoxModel();
    UColor color = option.getBorderColor();
    table.setBorderColor(new Color(color.getRed(), color.getGreen(), color.getBlue()));
    // Added to allow compilation
    Cell cell = null;
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
}