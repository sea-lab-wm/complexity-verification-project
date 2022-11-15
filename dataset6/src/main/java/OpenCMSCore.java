import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class OpenCMSCore {

    private CmsFileBuffer m_buffer = new CmsFileBuffer();
    private boolean m_exitCalled;
    private boolean m_interactive;
    private boolean m_echo;
    private OpenCMSWriter m_out;
    private PrintStream m_err;

    //ADDED BY KOBI
    public void runAll() {
        createContextMenu(new CmsSitemapHoverbar());
        try {
            seekFile(1, 1);
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        execute(new Reader() {
            @Override
            public void close() throws IOException {
                // TODO Auto-generated method stub
                
            }
            @Override
            public int read(char[] arg0, int arg1, int arg2) throws IOException {
                // TODO Auto-generated method stub
                return 0;
            }
        });
        try {
            generateContent(new CmsObject(), "vfsFolder", 1, 1);
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        buttonHtml(new CmsWorkplace());
    }

    // org.opencms.ade.sitemap.client.hoverbar.CmsHoverbarContextMenuButton.createContextMenu(org.opencms.ade.sitemap.client.hoverbar.CmsSitemapHoverbar)
    //SNIPPET_STARTS
    public List<A_CmsSitemapMenuEntry> createContextMenu(CmsSitemapHoverbar hoverbar) {

        List<A_CmsSitemapMenuEntry> result = Lists.newArrayList();

        result.add(new CmsGotoMenuEntry(hoverbar));
        result.add(new CmsGotoExplorerMenuEntry(hoverbar));
        result.add(new CmsOpenGalleryMenuEntry(hoverbar));
        result.add(new CmsEditRedirectMenuEntry(hoverbar));
        result.add(new CmsEditModelPageMenuEntry(hoverbar));
        result.add(new CmsDeleteModelPageMenuEntry(hoverbar));
        result.add(new CmsDisableMenuEntry(hoverbar));
        result.add(new CmsEditMenuEntry(hoverbar));
        result.add(new CmsCopyPageMenuEntry(hoverbar));
        result.add(new CmsCopyModelPageMenuEntry(hoverbar));
        result.add(new CmsSetDefaultModelMenuEntry(hoverbar));
        result.add(new CmsCopyAsModelGroupPageMenuEntry(hoverbar));
        result.add(new CmsCreateGalleryMenuEntry(hoverbar));
        result.add(new CmsResourceInfoMenuEntry(hoverbar));
        result.add(new CmsParentSitemapMenuEntry(hoverbar));
        result.add(new CmsGotoSubSitemapMenuEntry(hoverbar));
        result.add(new CmsNewChoiceMenuEntry(hoverbar));
        result.add(new CmsHideMenuEntry(hoverbar));
        result.add(new CmsShowMenuEntry(hoverbar));
        result.add(new CmsAddToNavMenuEntry(hoverbar));
        result.add(new CmsBumpDetailPageMenuEntry(hoverbar));
        result.add(new CmsRefreshMenuEntry(hoverbar));
        result.add(
                new CmsAdvancedSubmenu(
                        hoverbar,
                        Arrays.asList(
                                new CmsAvailabilityMenuEntry(hoverbar),
                                new CmsLockReportMenuEntry(hoverbar),
                                new CmsSeoMenuEntry(hoverbar),
                                new CmsSubSitemapMenuEntry(hoverbar),
                                new CmsMergeMenuEntry(hoverbar),
                                new CmsRemoveMenuEntry(hoverbar))));
        result.add(new CmsModelPageLockReportMenuEntry(hoverbar));
        result.add(new CmsDeleteMenuEntry(hoverbar));

        return result;
    }
    //SNIPPET_END
    // org.opencms.jlan.CmsJlanNetworkFile.seekFile(long,int)
//    @Override // Removed to allow compilation
    //SNIPPET_STARTS
    public long seekFile(long pos, int typ) throws IOException {

        try {
            load(true);
            switch (typ) {

                //  From current position

                case SeekType.CurrentPos:
                    m_buffer.seek(m_buffer.getPosition() + pos);
                    break;

                //  From end of file

                case SeekType.EndOfFile:
                    long newPos = m_buffer.getLength() + pos;
                    m_buffer.seek(newPos);
                    break;

                //  From start of file

                case SeekType.StartOfFile:
                default:
                    m_buffer.seek(pos);
                    break;
            }
            return m_buffer.getPosition();
        } catch (CmsException e) {
            throw new IOException(e);
        }
    }
    //SNIPPET_END
    // org.opencms.main.CmsShell.execute(java.io.Reader)
    //SNIPPET_STARTS
    /**
     * Executes the commands from the given reader in this shell.<p>
     *
     * <ul>
     * <li>Commands in the must be separated with a line break '\n'.
     * <li>Only one command per line is allowed.
     * <li>String parameters must be quoted like this: <code>'string value'</code>.
     * </ul>
     *
     * @param reader the reader from which the commands are read
     */
    public void execute(Reader reader) {

        try {
            LineNumberReader lnr = new LineNumberReader(reader);
            while (!m_exitCalled) {
                String line = lnr.readLine();
                if (line != null) {
                    if (m_interactive || m_echo) {
                        // print the prompt in front of the commands to process only when 'interactive'
                        printPrompt();
                    }
                } else {
                    // if null the file has been read to the end
                    try {
                        Thread.sleep(500);
                    } catch (Throwable t) {
                        // noop
                    }
                    // end the while loop
                    break;
                }
                if (line.trim().startsWith("#")) {
                    m_out.println(line);
                    continue;
                }
                StringReader lineReader = new StringReader(line);
                StreamTokenizer st = new StreamTokenizer(lineReader);
                st.eolIsSignificant(true);
                st.wordChars('*', '*');
                // put all tokens into a List
                List<String> parameters = new ArrayList<String>();
                while (st.nextToken() != StreamTokenizer.TT_EOF) {
                    if (st.ttype == StreamTokenizer.TT_NUMBER) {
                        parameters.add(Integer.toString(new Double(st.nval).intValue()));
                    } else {
                        parameters.add(st.sval);
                    }
                }
                lineReader.close();

                if (parameters.size() == 0) {
                    // empty line, just need to check if echo is on
                    if (m_echo) {
                        m_out.println();
                    }
                    continue;
                }

                // extract command and arguments
                String command = parameters.get(0);
                List<String> arguments = parameters.subList(1, parameters.size());

                // execute the command with the given arguments
                executeCommand(command, arguments);
            }
        } catch (Throwable t) {
            t.printStackTrace(m_err);
        }
    }
    //SNIPPET_END
    // org.opencms.test.OpenCmsTestCase.generateContent(org.opencms.file.CmsObject,java.lang.String,int,double)
    //SNIPPET_STARTS
    /**
     * Generates a sub tree of folders with files.<p>
     *
     * @param cms the cms context
     * @param vfsFolder name of the folder
     * @param numberOfFiles the number of files to generate
     * @param fileTypeDistribution a percentage: x% binary files and (1-x)% text files
     *
     * @return the number of files generated
     *
     * @throws Exception if something goes wrong
     */
    public static int generateContent(CmsObject cms, String vfsFolder, int numberOfFiles, double fileTypeDistribution)
            throws Exception {

        int maxProps = 10;
        double propertyDistribution = 0.0;
        int writtenFiles = 0;

        int numberOfBinaryFiles = (int)(numberOfFiles * fileTypeDistribution);

        // generate binary files
        writtenFiles += generateResources(
                cms,
                "org/opencms/search/pdf-test-112.pdf",
                vfsFolder,
                numberOfBinaryFiles,
                CmsResourceTypeBinary.getStaticTypeId(),
                maxProps,
                propertyDistribution);

        // generate text files
        writtenFiles += generateResources(
                cms,
                "org/opencms/search/extractors/test1.html",
                vfsFolder,
                numberOfFiles - numberOfBinaryFiles,
                CmsResourceTypePlain.getStaticTypeId(),
                maxProps,
                propertyDistribution);

        System.out.println("" + writtenFiles + " files written in Folder " + vfsFolder);

        return writtenFiles;
    }
    //SNIPPET_END
    // org.opencms.workplace.list.CmsListRadioMultiAction.buttonHtml(org.opencms.workplace.CmsWorkplace)
//    @Override // Removed to allow compilation
    //SNIPPET_STARTS
    public String buttonHtml(CmsWorkplace wp) {

        if (!isVisible()) {
            return "";
        }
        if (isEnabled()) {
            String onClic = "listRSelMAction('"
                    + getListId()
                    + "','"
                    + getId()
                    + "', '"
                    + CmsStringUtil.escapeJavaScript(wp.resolveMacros(getConfirmationMessage().key(wp.getLocale())))
                    + "', "
                    + CmsHtmlList.NO_SELECTION_MATCH_HELP_VAR
                    + getId()
                    + ", '"
                    + getRelatedActionIds()
                    + "');";
            return A_CmsHtmlIconButton.defaultButtonHtml(
                    CmsHtmlIconButtonStyleEnum.SMALL_ICON_TEXT,
                    getId(),
                    getName().key(wp.getLocale()),
                    getHelpText().key(wp.getLocale()),
                    isEnabled(),
                    getIconPath(),
                    null,
                    onClic);
        }
        return "";
    }
    //SNIPPET_END
    //SNIPPETS_END

    private void load(boolean b) {

    }

    private ConfirmationMessage getName() {
        return null;
    }

    private ConfirmationMessage getHelpText() {
        return null;
    }

    private Object getIconPath() {
        return null;
    }

    private String getRelatedActionIds() {
        return null;
    }

    private ConfirmationMessage getConfirmationMessage() {
        return null;
    }

    private String getId() {
        return null;
    }

    private String getListId() {
        return null;
    }

    private boolean isEnabled() {
        return false;
    }

    private boolean isVisible() {
        return false;
    }

    private static int generateResources(CmsObject cms, String s, String vfsFolder, int numberOfBinaryFiles, Object staticTypeId, int maxProps, double propertyDistribution) {
        return 0;
    }

    private void executeCommand(String command, List<String> arguments) {

    }

    private void printPrompt() {

    }

    private static class CmsObject {
    }

    private static class CmsResourceTypeBinary {
        public static Object getStaticTypeId() {
            return null;
        }
    }

    private static class CmsResourceTypePlain {
        public static Object getStaticTypeId() {
            return null;
        }
    }

    private class CmsSitemapHoverbar {
    }

    private class A_CmsSitemapMenuEntry {
    }

    private static class Lists {
        public static List<A_CmsSitemapMenuEntry> newArrayList() {
            return null;
        }
    }

    private class CmsGotoMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsGotoMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsGotoExplorerMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsGotoExplorerMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsOpenGalleryMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsOpenGalleryMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsEditRedirectMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsEditRedirectMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsEditModelPageMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsEditModelPageMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsDeleteModelPageMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsDeleteModelPageMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsDeleteMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsDeleteMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsModelPageLockReportMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsModelPageLockReportMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsRemoveMenuEntry {
        public CmsRemoveMenuEntry(CmsSitemapHoverbar hoverbar) {
        }
    }

    private class CmsMergeMenuEntry {
        public CmsMergeMenuEntry(CmsSitemapHoverbar hoverbar) {
        }
    }

    private class CmsSubSitemapMenuEntry {
        public CmsSubSitemapMenuEntry(CmsSitemapHoverbar hoverbar) {
        }
    }

    private class CmsSeoMenuEntry {
        public CmsSeoMenuEntry(CmsSitemapHoverbar hoverbar) {
        }
    }

    private class CmsLockReportMenuEntry {
        public CmsLockReportMenuEntry(CmsSitemapHoverbar hoverbar) {
        }
    }

    private class CmsAvailabilityMenuEntry {
        public CmsAvailabilityMenuEntry(CmsSitemapHoverbar hoverbar) {
        }
    }

    private class CmsAdvancedSubmenu extends A_CmsSitemapMenuEntry {
        public CmsAdvancedSubmenu(CmsSitemapHoverbar hoverbar, List<Object> objects) {
            super();
        }
    }

    private class CmsDisableMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsDisableMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsEditMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsEditMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsRefreshMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsRefreshMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsAddToNavMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsAddToNavMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsBumpDetailPageMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsBumpDetailPageMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsCopyPageMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsCopyPageMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsCopyModelPageMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsCopyModelPageMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsSetDefaultModelMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsSetDefaultModelMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsCopyAsModelGroupPageMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsCopyAsModelGroupPageMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsCreateGalleryMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsCreateGalleryMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsResourceInfoMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsResourceInfoMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsParentSitemapMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsParentSitemapMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsGotoSubSitemapMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsGotoSubSitemapMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsNewChoiceMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsNewChoiceMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsHideMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsHideMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    private class CmsShowMenuEntry extends A_CmsSitemapMenuEntry {
        public CmsShowMenuEntry(CmsSitemapHoverbar hoverbar) {
            super();
        }
    }

    public static class SeekType {
        public static final int CurrentPos = 1;
        public static final int EndOfFile = 0;
        public static final int StartOfFile = 2;
    }

    private class CmsFileBuffer {
        public long getPosition() throws CmsException{
            return 0;
        }

        public long getLength() {
            return 0;
        }

        public void seek(long pos) {

        }
    }

    private class CmsException extends IOException {
    }

    private class OpenCMSWriter {
        public void println(String line) {

        }

        public void println() {

        }
    }

    private class CmsWorkplace {
        public Object getLocale() {
            return null;
        }

        public Object resolveMacros(CmsWorkplace key) {
            return null;
        }
    }

    private class ConfirmationMessage {
        public CmsWorkplace key(Object locale) {
            return null;
        }
    }

    private static class CmsStringUtil {
        public static String escapeJavaScript(Object resolveMacros) {
            return null;
        }
    }

    private static class CmsHtmlList {
        public static final String NO_SELECTION_MATCH_HELP_VAR = "";
    }

    private static class A_CmsHtmlIconButton {
        public static String defaultButtonHtml(Object smallIconText, String id, CmsWorkplace key, CmsWorkplace key1, boolean enabled, Object iconPath, Object o, String onClic) {
            return null;
        }
    }

    private static class CmsHtmlIconButtonStyleEnum {
        public static final Object SMALL_ICON_TEXT = null;
    }
}
