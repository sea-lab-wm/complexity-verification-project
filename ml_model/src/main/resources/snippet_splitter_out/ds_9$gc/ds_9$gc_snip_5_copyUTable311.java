package snippet_splitter_out.ds_9$gc;

public class ds_9$gc_snip_5_copyUTable311 {
  // SNIPPET_END_3
  // Snippet 3
  // org.unicase.docExport.docWriter.ITextWriter.writeUTable
  // http://unicase.googlecode.com/svn/trunk/core/org.unicase.docExport/src/org/unicase/docExport/docWriter/
  // ITextWriter.java
  // S3_1:1 (Revision 1) method chains, good comments
  /** The images of a unified table are copied to a simplified “yes/no” table. */
  // SNIPPET_STARTS_1
  protected Table copyUTable311(UTable uTable) {
    /* Create a new table with the same size and color as uTable. */
    Table table = new Table(uTable.getColumnsCount());
    table.setBorderColor(
        new Color(
            uTable.getBoxModel().getBorderColor().getRed(),
            uTable.getBoxModel().getBorderColor().getGreen(),
            uTable.getBoxModel().getBorderColor().getBlue()));
    /* Go through all entries of uTable and copy image information. */
    // had to be added to allow compilation
    Cell cell = null;
    for (UTableCell uCell : uTable.getEntries()) {
      UParagraph paragraph = uCell.getContent();
      /* Check if there is image information to copy. */
      if (paragraph.getChildren().size() > 0 && paragraph.getChildren().get(0) instanceof UImage) {
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
}
