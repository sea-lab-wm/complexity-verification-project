package snippet_splitter_out.ds_9$bc;

public class ds_9$bc_snip_5_copyUTable312 {
  // SNIPPET_END_1
  // S3_1:2 (Revision 1) method chains, bad comments
  /** The images of a unified table are copied to a simplified “yes/no” table. */
  // SNIPPET_STARTS_2
  protected Table copyUTable312(UTable uTable) {
    /* Define local variables. */
    Table table = new Table(uTable.getColumnsCount());
    table.setBorderColor(
        new Color(
            uTable.getBoxModel().getBorderColor().getRed(),
            uTable.getBoxModel().getBorderColor().getGreen(),
            uTable.getBoxModel().getBorderColor().getBlue()));
    // Had to be added to allow compilation
    Cell cell = null;
    for (UTableCell uCell : uTable.getEntries()) {
      /* Define local variable. */
      UParagraph paragraph = uCell.getContent();
      /* If children size larger than 0 and first children is an image. */
      if (paragraph.getChildren().size() > 0 && paragraph.getChildren().get(0) instanceof UImage) {
        UImage image = paragraph.getChildren().get(0);
        /* If segment starts with false, then create a cell with “no”. */
        /* Otherwise, if segment starts with true, then a create cell */
        /* with “yes”. Otherwise create cell with segment. */
        if (image.getPath().lastSegment().startsWith("false")) {
          cell = new Cell("no");
        } else {
          if (image.getPath().lastSegment().startsWith("true")) {
            cell = new Cell("yes");
          } else {
            cell = new Cell(image.getPath().lastSegment());
          }
        }
        /* Copy background color. */
        if (uCell.getBoxModel().getBackgroundColor() != null) {
          cell.setBackgroundColor(uCell.getBoxModel().getBackgroundColor());
        }
      }
      table.addCell(cell);
    }
    return table;
  }
}
