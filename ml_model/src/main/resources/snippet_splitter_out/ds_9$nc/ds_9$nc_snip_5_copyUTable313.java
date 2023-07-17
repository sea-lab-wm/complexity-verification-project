package snippet_splitter_out.ds_9$nc;

public class ds_9$nc_snip_5_copyUTable313 {
  // SNIPPET_END_2
  // S3_1:3 (Revision 1) method chains, no comments
  /** The images of a unified table are copied to a simplified “yes/no” table. */
  // SNIPPET_STARTS_3
  protected Table copyUTable313(UTable uTable) {
    Table table = new Table(uTable.getColumnsCount());
    table.setBorderColor(
        new Color(
            uTable.getBoxModel().getBorderColor().getRed(),
            uTable.getBoxModel().getBorderColor().getGreen(),
            uTable.getBoxModel().getBorderColor().getBlue()));
    // Had to be added to allow compilation
    Cell cell = null;
    for (UTableCell uCell : uTable.getEntries()) {
      UParagraph paragraph = uCell.getContent();
      if (paragraph.getChildren().size() > 0 && paragraph.getChildren().get(0) instanceof UImage) {
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
}
