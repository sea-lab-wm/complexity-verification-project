package cog_complexity_validation_datasets.Three;

import javax.script.Bindings;
import javax.script.ScriptEngine;
import javax.script.ScriptException;
import javax.swing.*;

import java.awt.*;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.beans.Expression;
import java.io.*;
import java.sql.SQLException;
import java.util.*;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.logging.*;     //CHANGED BY KOBI
import java.util.Map.Entry;     //ADDED BY KOBI

/**
 * Note: method names were changed from 'main' to main + their task number
 *       this allows the code to be compiled and analyzed by sonarqube
 */
public class Tasks_3 {

    @SuppressWarnings("all")
    private static final String RETURN = "";
    @SuppressWarnings("all")
    private static final int IRETURN = 0;
    @SuppressWarnings("all")
    private static final Object BUTTON_IMAGE = null;
    @SuppressWarnings("all")
    private static final Object BUTTON_ROLLOVER_IMAGE = null;
    @SuppressWarnings("all")
    private static final int EQ = 0;
    @SuppressWarnings("all")
    private static final int NE = 1;
    @SuppressWarnings("all")
    private static final int COLUMNS = 0;
    @SuppressWarnings("all")
    private static final int DEFAULT_WIDTH = 1;
    @SuppressWarnings("all")
    private static final int DEFAULT_HEIGHT = 2;
    @SuppressWarnings("all")
    private static final Locale CONSTRUCTOR_ERROR_FORMAT = null;
    @SuppressWarnings("all")
    private static final int ACC_PUBLIC = 0;
    @SuppressWarnings("all")
    private static final Object MESSAGE_COUNT = null;
    @SuppressWarnings("all")
    private static final int DATABASE_CLOSING = 0;
    @SuppressWarnings("all")
    private static final int DATABASE_ONLINE = 1;
    @SuppressWarnings("all")
    private static final Object CHANNELLIST = null;
    @SuppressWarnings("all")
    private static final boolean COLUMN = false;
    @SuppressWarnings("all")
    private static final boolean VALUE = false;
    @SuppressWarnings("all")
    private static final boolean FUNCTION = false;
    @SuppressWarnings("all")
    private static final boolean ALTERNATIVE = false;
    @SuppressWarnings("all")
    private static final boolean CASEWHEN = false;
    @SuppressWarnings("all")
    private static final boolean CONVERT = false;
    @SuppressWarnings("all")
    private static ScriptEngine body;
    @SuppressWarnings("all")
    private static String callstack;
    @SuppressWarnings("all")
    private static Bindings interpreter;
    @SuppressWarnings("all")
    private static LinkedList<ActionMenu> actionList;
    @SuppressWarnings("all")
    private static IdentityHashMap<Object, Object> databaseIDMap;
    @SuppressWarnings("all")
    private static Spring option;
    @SuppressWarnings("all")
    private static JSpinner spinner;
    @SuppressWarnings("all")
    private static Object tempCalDefault;
    @SuppressWarnings("all")
    private static int[] jjbitVec0;
    @SuppressWarnings("all")
    private static int[] jjbitVec1;
    @SuppressWarnings("all")
    private int fParameterSetNumber;
    @SuppressWarnings("all")
    private Object fParameters = null;
    @SuppressWarnings("all")
    private int constType;
    @SuppressWarnings("all")
    private ConstraintCore core = new ConstraintCore();
    @SuppressWarnings("all")
    private HsqlName constName = new HsqlName();
    @SuppressWarnings("all")
    private String DELTA_START;
    @SuppressWarnings("all")
    private int fPrefix;
    @SuppressWarnings("all")
    private int fSuffix;
    @SuppressWarnings("all")
    private String DELTA_END;
    @SuppressWarnings("all")
    private boolean windowed;
    @SuppressWarnings("all")
    private GraphicsDevice gd;
    @SuppressWarnings("all")
    private Object xsp;
    @SuppressWarnings("all")
    private Object jj_scanpos;
    @SuppressWarnings("all")
    private IntValueHashMap rightsMap;
    @SuppressWarnings("all")
    private String granteeName;
    @SuppressWarnings("all")
    private GranteeManager granteeManager;
    @SuppressWarnings("all")
    private Ftest fTest;
    @SuppressWarnings("all")
    private Method fMethod;
    @SuppressWarnings("all")
    private Parent parent;
    private InGameController inGameController;
    @SuppressWarnings("all")
    private Object clas;
    @SuppressWarnings("all")
    private String value;
    @SuppressWarnings("all")
    private Object asClass;
    @SuppressWarnings("all")
    private BtPanel btPanel;
    @SuppressWarnings("all")
    private Object cancel;
    @SuppressWarnings("all")
    private Object ok;
    @SuppressWarnings("all")
    private BtPanel panel;
    @SuppressWarnings("all")
    private List<Object> nameList;
    @SuppressWarnings("all")
    private Object sourceFileInfo;
    @SuppressWarnings("all")
    private int num;
    @SuppressWarnings("all")
    private Vector<DeviceIf> mDevices;
    @SuppressWarnings("all")
    private In in;
    @SuppressWarnings("all")
    private boolean active;
    @SuppressWarnings("all")
    private int ON;
    @SuppressWarnings("all")
    private int OFF;
    @SuppressWarnings("all")
    private Object returnType;
    @SuppressWarnings("all")
    private Object mCloneData;
    @SuppressWarnings("all")
    private Object mOwner;
    @SuppressWarnings("all")
    private CapturePluginPanel mPanel;
    @SuppressWarnings("all")
    private Object mCurrentPanel;
    @SuppressWarnings("all")
    private Object player;
    @SuppressWarnings("all")
    private Settlement inSettlement;
    @SuppressWarnings("all")
    private BtPanel destinations;
    @SuppressWarnings("all")
    private int len;
    @SuppressWarnings("all")
    private int bufpos;
    @SuppressWarnings("all")
    private Object ret;
    @SuppressWarnings("all")
    private int bufsize;
    @SuppressWarnings("all")
    private Object buffer;
    @SuppressWarnings("all")
    private Vector<Runner> fRunners;
    @SuppressWarnings("all")
    private OptionalDataException parameters;
    @SuppressWarnings("all")
    private int outlen;
    @SuppressWarnings("all")
    private int offset;
    @SuppressWarnings("all")
    private MapTransform currentMapTransform;
    @SuppressWarnings("all")
    private FreeColClient freeColClient;
    @SuppressWarnings("all")
    private Object iterateOverMe;
    @SuppressWarnings("all")
    private int mYear;
    @SuppressWarnings("all")
    private int mMonth;
    @SuppressWarnings("all")
    private int fContextLength;
    @SuppressWarnings("all")
    private String fExpected;
    @SuppressWarnings("all")
    private String fActual;
    @SuppressWarnings("all")
    private Object cs;
    @SuppressWarnings("all")
    private Object object;
    @SuppressWarnings("all")
    private Object name;
    @SuppressWarnings("all")
    private List<AbstractPluginProgramFormating> mConfigs;
    @SuppressWarnings("all")
    private BtPanel list;
    @SuppressWarnings("all")
    private Package method;
    @SuppressWarnings("all")
    private Boolean[] row;
    @SuppressWarnings("all")
    private Ns ns;
    @SuppressWarnings("all")
    private Object defschema;
    @SuppressWarnings("all")
    private Object schema;
    @SuppressWarnings("all")
    private Ns t;
    @SuppressWarnings("all")
    private int jj_ntk;
    @SuppressWarnings("all")
    private ArrayList mChildNodes;
    @SuppressWarnings("all")
    private Object mMarker;
    @SuppressWarnings("all")
    private Interval next;
    @SuppressWarnings("all")
    private int mNumber;
    @SuppressWarnings("all")
    private Object mName;
    @SuppressWarnings("all")
    private boolean isClosed;
    @SuppressWarnings("all")
    private Result resultOut;
    @SuppressWarnings("all")
    private boolean blinkOn;
    @SuppressWarnings("all")
    private Cc cc;
    @SuppressWarnings("all")
    private Cc mLocalizer;
    @SuppressWarnings("all")
    private Unit UiUtilities;
    @SuppressWarnings("all")
    private Object date;
    @SuppressWarnings("all")
    private TimeDateChooserPanel mTimePanel;
    @SuppressWarnings("all")
    private Klass klass;
    @SuppressWarnings("all")
    private SuiteMethod suiteMethod;
    @SuppressWarnings("all")
    private Test suite;
    @SuppressWarnings("all")
    private Object missionChip;
    @SuppressWarnings("all")
    private boolean expertMission;
    @SuppressWarnings("all")
    private Object color;
    @SuppressWarnings("all")
    private MissionChip expertMissionChips;
    @SuppressWarnings("all")
    private Stack stack;
    @SuppressWarnings("all")
    private Class<Object> runnerClass;
    @SuppressWarnings("all")
    private Object fTestClass;
    @SuppressWarnings("all")
    private boolean isReadOnly;
    @SuppressWarnings("all")
    private WareHouse warehouseDialog;
    @SuppressWarnings("all")
    private Object FRETURN;
    @SuppressWarnings("all")
    private Object opcode;
    @SuppressWarnings("all")
    private Object LRETURN;
    @SuppressWarnings("all")
    private Cv cv;
    @SuppressWarnings("all")
    private int version;
    @SuppressWarnings("all")
    private String dataServiceId;
    @SuppressWarnings("all")
    private List<Method> fTestMethods;
    @SuppressWarnings("all")
    private Iterator<Object> classNames;
    @SuppressWarnings("all")
    private String clsName;
    @SuppressWarnings("all")
    private Boolean clsCat;
    @SuppressWarnings("all")
    private Object clsSchem;
    @SuppressWarnings("all")
    private Message[] messages;
    @SuppressWarnings("all")
    private ProgramTable mProgramTable;
    @SuppressWarnings("all")
    private ProgramTable mProgramTableModel;
    @SuppressWarnings("all")
    private ProgramTable devplugin;
    @SuppressWarnings("all")
    private Tc tc;
    @SuppressWarnings("all")
    private Connection session;
    @SuppressWarnings("all")
    private JInternalFrame f;
    @SuppressWarnings("all")
    private Point loc;
    @SuppressWarnings("all")
    private ArrayDeque<Object> classNameSet;
    @SuppressWarnings("all")
    private String className;
    @SuppressWarnings("all")
    private Object andAliases;
    @SuppressWarnings("all")
    private Object methods;
    @SuppressWarnings("all")
    private Device device;
    @SuppressWarnings("all")
    private ObjectOutput out;
    @SuppressWarnings("all")
    private Object mCount;
    @SuppressWarnings("all")
    private Device dev;
    @SuppressWarnings("all")
    private int fieldcount;
    @SuppressWarnings("all")
    private int[] cols;
    @SuppressWarnings("all")
    private int[] a;
    @SuppressWarnings("all")
    private int[] coltypes;
    @SuppressWarnings("all")
    private int[] b;
    @SuppressWarnings("all")
    private Vector importedPackages;
    @SuppressWarnings("all")
    private String cmpDataServiceId;
    @SuppressWarnings("all")
    private Tasks_3 cmp;
    @SuppressWarnings("all")
    private ConstraintCore mainCols;
    @SuppressWarnings("all")
    private Object refCols;
    @SuppressWarnings("all")
    private Messagee messages2;
    @SuppressWarnings("all")
    private Object fMessage;
    @SuppressWarnings("all")
    private int currentMode;
    @SuppressWarnings("all")
    private Gui gui;
    @SuppressWarnings("all")
    private DataInput input_stream;
    @SuppressWarnings("all")
    private char curChar;
    @SuppressWarnings("all")
    private Script mAppleScript;
    @SuppressWarnings("all")
    private Fklass fKlass;
    @SuppressWarnings("all")
    private Node x;
    @SuppressWarnings("all")
    private Count ts;
    @SuppressWarnings("all")
    private Mexpression expression;
    @SuppressWarnings("all")
    private String ddl;
    @SuppressWarnings("all")
    private HashMap<String, Integer> messagesToIgnore;

    //ADDED BY KOBI
    @SuppressWarnings("all")
    private Logger logger;

    @SuppressWarnings("all")
    public Tasks_3(Class<?> klass) {

    }

    @SuppressWarnings("all")
    public Tasks_3(String message) {

    }

    //ADDED BY KOBI
    @SuppressWarnings("all")
    public void runAllSnippets() {
        //S71
        getASMModifiers(new Modifiers());
        //S72
        getListCellRendererComponent(new JList<>(), new Object(), 1, true, true);
        //S73
        compact("message");
        //S74
        s74();
        //S75
        loadColorChip(GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice().getDefaultConfiguration(), new Color(1));
        //S76
        try {
            s76();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S77
        try {
            s77();
        } catch (FileNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S78
        //SKIPPED!!!!!!!!!!!!!!
        //S79
        s79();
        //S80
        returnToTitle();
        //S81
        importPackage("name");
        //S82
        s82();
        //S83
        try {
            filter2(new Filter());
        } catch (NoTestsRemainException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S84
        s84();
        //S85
        addMessage(new GUIMessage());
        //S86
        jjCanMove_1(1, 1, 1, 1, 1);
        //S87
        correctTimeZone(new Date());
        //S88
        getMessage();
        //S89
        getStateString();
        //S90
        displayTileCursor(new Tile(), 1, 1);
        //S91
        jjMoveStringLiteralDfa18_0(1, 1, 1, 1);
        //S92
        getAvailableChannels();
        //S93
        try {
            getParametersMethod();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S94
        s94();
        //S95
        s95();
        //S96
        actionPerformed2(new ActionEvent(new Tasks_3("message"), 1, "blink"));
        //S97
        getButtonAction();
        //S98
        s98();
        //S99
        s99();
        //S100
        purgeOldMessagesFromMessagesToIgnore(1);
    }

    // Snippet s71
    /**
     Translate bsh.Modifiers into ASM modifier bitflags.
     */
    //SNIPPET_STARTS
    static int getASMModifiers( Modifiers modifiers )
    {
        int mods = 0;
        if ( modifiers == null )
            return mods;

        if ( modifiers.hasModifier("public") )
            mods += ACC_PUBLIC;
        return 0; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s72
    public class Component extends Tasks_3{ // Added class wrapper to allow compilation
        public Component(String message) {
            super(message);
        }

        //SNIPPET_STARTS
        public Component getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {

            JLabel label = (JLabel) super.getListCellRendererComponent(list, value, index, isSelected, cellHasFocus);

            String str;

            if (value instanceof DeviceIf) {
                DeviceIf device = (DeviceIf)value;

            } // Added to allow compilation
            return new Component(str = "");                                                /*Altered return*/
            //return null; // Added to allow compilation
        } // Added to allow compilation
    }

    // Snippet s73
    //SNIPPET_STARTS
    public String compact(String message) {
        if (fExpected == null || fActual == null || areStringsEqual())
            return Assert.format(message, fExpected, fActual);

        findCommonPrefix();
        findCommonSuffix();

        return message;                                                                 /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s74
    //SNIPPET_STARTS
    public void s74() {
        classNames = classNameSet.iterator();

        while (classNames.hasNext()) {
            className = (String) classNames.next();
            methods = iterateRoutineMethods(className, andAliases);
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s75
    /**
     * Generates a color chip image and stores it in memory.
     *
     * @param gc The GraphicsConfiguration is needed to create images that are
     *            compatible with the local environment.
     * @param c The color of the color chip to create.
     */
    //SNIPPET_STARTS
    private void loadColorChip(GraphicsConfiguration gc, Color c) {
        BufferedImage tempImage = gc.createCompatibleImage(11, 17);
        Graphics g = tempImage.getGraphics();
        if (c.equals(Color.BLACK)) {
            g.setColor(Color.WHITE);
        }
    } // Added to allow compilation

    // Snippet s76
    //SNIPPET_STARTS
    public void s76() throws IOException{
        out.writeObject(device.getDriver().getClass().getName());
        out.writeObject(device.getName());

        device.writeData(out);
    } // Added to allow compilation

    // Snippet s77
    //SNIPPET_STARTS
    public void s77() throws FileNotFoundException, IOException{
        File data = new File(Plugin.getPluginManager().getTvBrowserSettings().getTvBrowserUserHome()  + File.separator +
                "CaptureDevices" + File.separator + mCount + ".dat");

        ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(data));

        dev.writeData(stream);
    } // Added to allow compilation

    // Snippet s78                                                                                      /*ORIGINALLY COMMENTED OUT*/
    //SNIPPET_STARTS
    private static Class<?>[] getAnnotatedClasses(Class<?> klass) throws InitializationError {
            SuiteClasses annotation= klass.getAnnotation(SuiteClasses.class);
            if (annotation == null)
                throw new Tasks_3("message").new InitializationError(String.format("class '%s' must have a SuiteClasses annotation", klass.getName()));
            return annotation.value();
    } // Added to allow compilation

    // Snippet s79
    //SNIPPET_STARTS
    public int s79() {
        for (int j = 0; j < fieldcount; j++) {
            int i = Column.compare(session.database.collation, a[cols[j]],
                    b[cols[j]], coltypes[cols[j]]);

            if (i != 0) {
                return i;
            }
        }

        return 0;
    } // Added to allow compilation

    // Snippet s80
    /**
     * Closes all panels, changes the background and shows the main menu.
     */
    //SNIPPET_STARTS
    public void returnToTitle() {
        // TODO: check if the GUI object knows that we're not
        // inGame. (Retrieve value of GUI::inGame.)  If GUI thinks
        // we're still in the game then log an error because at this
        // point the GUI should have been informed.
        closeMenus();
        removeInGameComponents();
        showMainPanel();
    } // Added to allow compilation

    // Snippet s81
    /**
     subsequent imports override earlier ones
     */
    //SNIPPET_STARTS
    public void	importPackage(String name)
    {
        if(importedPackages == null)
            importedPackages = new Vector();

        // If it exists, remove it and add it at the end (avoid memory leak)
        if ( importedPackages.contains( name ) )
            importedPackages.remove( name );

        importedPackages.addElement(name);
    } // Added to allow compilation

    // Snippet s82
    //SNIPPET_STARTS
    public boolean s82() {
        if(dataServiceId.compareTo(cmpDataServiceId) != 0) {
            return false;
        }

        String country = getCountry();
        String cmpCountry = cmp.getCountry();

        return false; // Added to allow compilation
    } // Added to allow compilation

    private String getCountry() {
        return new String();                                            /*Altered return*/
        //return null;
    }

    // Snippet s83
    //SNIPPET_STARTS
    public void filter2(Filter filter) throws NoTestsRemainException { // Renamed to allow compilation
        for (Iterator<Method> iter= fTestMethods.iterator(); iter.hasNext();) {
            Method method= iter.next();
            if (!filter.shouldRun(methodDescription(method)))
                iter.remove();
        }
        if (fTestMethods.isEmpty())
            throw new NoTestsRemainException();
    } // Added to allow compilation

    // Snippet s84
    //SNIPPET_STARTS
    public void s84() {
        /* fredt - in FK constraints column lists for iColMain and iColRef have
           identical sets to visible columns of iMain and iRef respectively
           but the order of columns can be different and must be preserved
         */
        core.mainColArray = mainCols;
        core.colLen       = core.mainColArray.length;
        core.refColArray  = refCols;
    } // Added to allow compilation

    // Snippet s85
    /**
     * Adds a message to the list of messages that need to be displayed on the GUI.
     * @param message The message to add.
     */
    //SNIPPET_STARTS
    public synchronized void addMessage(GUIMessage message) {
        if (getMessageCount() == MESSAGE_COUNT) {
            messages2.remove(0); // Renamed to allow compilation
        }
        messages2.add(message); // Renamed to allow compilation

        freeColClient.getCanvas().repaint(0, 0, getWidth(), getHeight());
    } // Added to allow compilation

    // Snippet s86
    //SNIPPET_STARTS
    private static final boolean jjCanMove_1(int hiByte, int i1, int i2, long l1, long l2)
    {
        switch(hiByte) {
            case 0:
                return ((jjbitVec0[i2] & l2) != 0L);
            default:
                if ((jjbitVec1[i1] & l1) != 0L)
                    return true;
                return false;
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s87
    //SNIPPET_STARTS
    private static Date correctTimeZone(final Date date) {
        Date ret=date;
        if(java.util.TimeZone.getDefault().useDaylightTime()){
            if(java.util.TimeZone.getDefault().inDaylightTime(date))
                ret.setTime(date.getTime()+1*60*60*1000);
        }
        return ret;
    } // Added to allow compilation

    // Snippet s88
//    @Override // Remvoed to allow compilation
    //SNIPPET_STARTS
    public String getMessage() {
        StringBuilder builder= new StringBuilder();
        if (fMessage != null)
            builder.append(fMessage);
        builder.append("arrays first differed at element ");

        return new String();                                                            /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s89
    //SNIPPET_STARTS
    String getStateString() {

        int state = getState();

        switch (state) {

            case DATABASE_CLOSING:
                return "DATABASE_CLOSING";

            case DATABASE_ONLINE:
                return "DATABASE_ONLINE";
        } // Added to allow compilation
        return new String();                                                           /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s90
    //SNIPPET_STARTS
    public boolean displayTileCursor(Tile tile, int canvasX, int canvasY) {
        if (currentMode == ViewMode.VIEW_TERRAIN_MODE) {

            Position selectedTilePos = gui.getSelectedTile();
            if (selectedTilePos == null)
                return false;

            if (selectedTilePos.getX() == tile.getX() && selectedTilePos.getY() == tile.getY()) {
                TerrainCursor cursor = gui.getCursor();
            } // Added to allow compilation
        } // Added to allow compilation
        return false; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s91
    //SNIPPET_STARTS
    private final int jjMoveStringLiteralDfa18_0(long old1, long active1, long old2, long active2)
    {
        if (((active1 &= old1) | (active2 &= old2)) == 0L)
            return jjStartNfa_0(16, 0L, old1, old2);
        try { curChar = input_stream.readChar(); }
        catch(java.io.IOException e) {
            jjStopStringLiteralDfa_0(17, 0L, active1, active2);
        }
        return 0; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s92
    /**
     * Get the List of all available Channels
     *
     * @return All available Channels
     */
    //SNIPPET_STARTS
    public ElgatoChannel[] getAvailableChannels() {
        ArrayList<ElgatoChannel> list = new ArrayList<ElgatoChannel>();

        String res = null;
        try {
            res = mAppleScript.executeScript(CHANNELLIST);
        } finally {
            // Added to allow compilation
        }
        return new ElgatoChannel[0];                                                /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s93                                                                  /*ORIGINALLY COMMENTED OUT*/
    //SNIPPET_STARTS
    private Method getParametersMethod() throws Exception {
        for (Method each : fKlass.getMethods()) {
            if (Modifier.isStatic(each.getModifiers())) {
                Annotation[] annotations= each.getAnnotations();
                for (Annotation annotation : annotations) {
                    if (annotation.annotationType().getClass() == Parameters.class) //.getClass() ADDED BY KOBI
                        return each;
                }
            }
        }
        throw new Exception("No public static parameters method on class "
                + getName());
    } // Added to allow compilation

    // Snippet s94
    //SNIPPET_STARTS
    public void s94() {
        Node r = x.getRight();

        if (r != null) {
            x = r;

            Node l = x.getLeft();
        }
    } // Added to allow compilation

    // Snippet s95
    //SNIPPET_STARTS
    public void s95() {
        InGameInputHandler inGameInputHandler = freeColClient.getInGameInputHandler();

        freeColClient.getClient().setMessageHandler(inGameInputHandler);
        gui.setInGame(true);
    } // Added to allow compilation

    // Snippet s96
    /**
     * Applies this action.
     *
     * @param e The <code>ActionEvent</code>.
     */
    //SNIPPET_STARTS
    public void actionPerformed2(ActionEvent e) { // Renamed to allow compilation
        final Game game = freeColClient.getGame();
        final Map map = game.getMap();

        Parameters p = showParametersDialog();
    } // Added to allow compilation

    // Snippet s97
    //SNIPPET_STARTS
    public ActionMenu getButtonAction() {
        AbstractAction action = new AbstractAction() {

            public void actionPerformed(ActionEvent evt) {
                showDialog();
            }
        };
        action.putValue(Action.NAME, mLocalizer.msg("CapturePlugin", "Capture Plugin"));
        action.putValue(Action.SMALL_ICON, createImageIcon("mimetypes", "video-x-generic", 16));

        return new ActionMenu();                                                                                                    /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s98
    //SNIPPET_STARTS
    public void s98() {
        Description description= Description.createSuiteDescription(name);
        int n= ts.testCount();
        for (int i= 0; i < n; i++)
            description.addChild(makeDescription(ts.testAt(i)));
    } // Added to allow compilation

    // Snippet s99
    //SNIPPET_STARTS
    public Object s99() {
        if (expression.exprType != VALUE && expression.exprType != COLUMN
                && expression.exprType != FUNCTION
                && expression.exprType != ALTERNATIVE
                && expression.exprType != CASEWHEN
                && expression.exprType != CONVERT) {
            StringBuffer temp = new StringBuffer();

            ddl = temp.append('(').append(ddl).append(')').toString();
        }

        return ddl;
    } // Added to allow compilation

    // Snippet s100                                                                     /*ORGINALLY COMMENTED OUT*/
    //SNIPPET_STARTS
    private synchronized void purgeOldMessagesFromMessagesToIgnore(int thisTurn) {
        List<String> keysToRemove = new ArrayList<String>();
        for (Entry<String, Integer> entry : messagesToIgnore.entrySet()) {
            if (entry.getValue().intValue() < thisTurn - 1) {
                if (logger.isLoggable(Level.FINER)) {
                    logger.finer("Removing old model message with key " + entry.getKey() + " from ignored messages.");
                }
                keysToRemove.add(entry.getKey());
            }
        }
    } // Added to allow compilation
    //SNIPPETS_END

    //ADDED BY KOBI
    @SuppressWarnings("all")
    public @interface SuiteClasses {
        Class<?>[] value();
    }

    //ADDED BY KOBI
    @SuppressWarnings("all")
    private class None {

    }

    @SuppressWarnings("all")
    private static long getTimeInMillis(Object tempCalDefault) {
        return 0;
    }

    @SuppressWarnings("all")
    private static void resetToTime(Object tempCalDefault) {

    }

    @SuppressWarnings("all")
    private static void setTimeInMillis(Object tempCalDefault, long t) {

    }

    @SuppressWarnings("all")
    private PrintWriter getWriter() {
        return null;
    }

    @SuppressWarnings("all")
    private boolean isPrimitive(Object returnType) {
        return false;
    }

    @SuppressWarnings("all")
    private void fireActionEvent(ActionEvent blinkEvent) {

    }

    @SuppressWarnings("all")
    private void setStrictJava(Object strictJava) {

    }

    @SuppressWarnings("all")
    private class Option {

    }

    @SuppressWarnings("all")
    private static boolean isEquals(Object expected, Object actual) {
        return false;
    }

    @SuppressWarnings("all")
    private class BtPanel {
        @SuppressWarnings("all")
        public void add(Object cancel) {
        }

        @SuppressWarnings("all")
        public void add(BtPanel btPanel, String borderLayout) {

        }

        @SuppressWarnings("all")
        public void add(TimeDateChooserPanel timeDateChooserPanel, String s) {

        }

        @SuppressWarnings("all")
        public boolean isEmpty() {
            return false;
        }

        @SuppressWarnings("all")
        public int size() {
            return 0;
        }

        @SuppressWarnings("all")
        public Object toArray(ProgramReceiveTarget[] programReceiveTargets) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class RootPane{

        @SuppressWarnings("all")
        public void setDefaultButton(Object ok) {

        }
    }

    @SuppressWarnings("all")
    private RootPane getRootPane() {
        RootPane rp = new RootPane();
        return rp;
    }

    @SuppressWarnings("all")
    private class InGameController {
        public void moveActiveUnit(Object sw) {
        }
    }

    @SuppressWarnings("all")
    private class Parent {

        @SuppressWarnings("all")
        public boolean isMapboardActionsEnabled() {
            return false;
        }

        @SuppressWarnings("all")
        public Object getStrictJava() {
            return null;
        }

        @SuppressWarnings("all")
        public This getGlobal(Interpreter declaringInterpreter) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private void super1(Class<?> aClass, Class<Before> beforeClass, Class<After> afterClass, Object test) {

    }

    @SuppressWarnings("all")
    private boolean jj_scan_token(int i) {
        return false;
    }

    @SuppressWarnings("all")
    private ConnectController getConnectController() {
        return null;
    }

    @SuppressWarnings("all")
    private String computeCommonSuffix() {
        return null;
    }

    @SuppressWarnings("all")
    private String computeCommonPrefix() {
        return null;
    }

    @SuppressWarnings("all")
    private static class ReturnControl {
        public String kind;
    }

    @SuppressWarnings("all")
    private static class ActionMenu {
        @SuppressWarnings("all")
        public AbstractButton[] getSubItems() {
            return new AbstractButton[0];
        }
    }

    @SuppressWarnings("all")
    private static class GranteeManager {
    }

    @SuppressWarnings("all")
    public static class HsqlException extends Exception {
    }

    @SuppressWarnings("all")
    private static class Database {
        public Object collation;
    }

    @SuppressWarnings("all")
    private class ConnectController {
        @SuppressWarnings("all")
        public void quitGame(boolean b) {

        }
    }

    @SuppressWarnings("all")
    private class Grantee {
    }

    @SuppressWarnings("all")
    private class IntValueHashMap {
    }

    @SuppressWarnings("all")
    private class Method {
        @SuppressWarnings("all")
        public Object getModifiers() {
            return null;
        }

        @SuppressWarnings("all")
        public Annotation[] getAnnotations() {
            return new Annotation[0];
        }

        //ADDED BY KOBI
        @SuppressWarnings("all")
        public Test getAnnotation(Class <?> c) {
            return new Test();
        }
    }

    @SuppressWarnings("all")
    private class RunNotifier {
    }

    @SuppressWarnings("all")
    private static class Description {
        @SuppressWarnings("all")
        public static Description createSuiteDescription(Object name) {
            return null;
        }

        @SuppressWarnings("all")
        public void addChild(Object methodDescription) {

        }
    }

    @SuppressWarnings("all")
    private class Before {
    }

    @SuppressWarnings("all")
    private class After {
    }

    @SuppressWarnings("all")
    private static class BshClassManager {
        @SuppressWarnings("all")
        public static BshClassManager createClassManager(Tasks_3 tasks) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class DeviceIf {
    }

    @SuppressWarnings("all")
    private class DeviceFileHandling {
    }

    @SuppressWarnings("all")
    private class In {
        @SuppressWarnings("all")
        public Object readObject() {
            return null;
        }

        @SuppressWarnings("all")
        public String readInt() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private static class Version {
        @SuppressWarnings("all")
        public static String id() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class Failure {
    }

    @SuppressWarnings("all")
    private class HsqlName {
    }

    @SuppressWarnings("all")
    private class Table {
    }

    @SuppressWarnings("all")
    private class ConstraintCore {
        public ConstraintCore mainColArray;
        public Object colLen;
        public Object length;
        public Object refColArray;
    }

    @SuppressWarnings("all")
    private class ActionEvent {
        @SuppressWarnings("all")
        public ActionEvent(Tasks_3 tasks, int eventId, String blink) {
        }
    }

    @SuppressWarnings("all")
    private class CapturePluginPanel {
        @SuppressWarnings("all")
        public CapturePluginPanel(Object mOwner, Object mCloneData) {
        }

        @SuppressWarnings("all")
        public void setBorder(Object emptyBorder) {

        }

        @SuppressWarnings("all")
        public void setSelectedTab(Object mCurrentPanel) {

        }
    }

    @SuppressWarnings("all")
    private static class Borders {
        @SuppressWarnings("all")
        public static Object createEmptyBorder(Object dluy5, Object dlux5, Object dluy51, Object dlux51) {
            return null;
        }
    }

    private class Result {
        @SuppressWarnings("all")
        public int getFailureCount() {
            return 0;
        }

        @SuppressWarnings("all")
        public void setResultType(int sqldisconnect) {

        }
    }

    @SuppressWarnings("all")
    private class PathNode {
        @SuppressWarnings("all")
        public Tile getTile() {
            return null;
        }

        @SuppressWarnings("all")
        public int getTurns() {
            return 0;
        }
    }

    @SuppressWarnings("all")
    private static class Unit {
        @SuppressWarnings("all")
        public static Object getXMLElementTagName() {
            return null;
        }

        @SuppressWarnings("all")
        public Object getFreeColGameObject(Object id) {
            return null;
        }

        @SuppressWarnings("all")
        public void readFromXMLElement(Element unitElement) {

        }

        @SuppressWarnings("all")
        public BtPanel createHelpTextArea(Object msg) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class Tile {
        @SuppressWarnings("all")
        public Settlement getSettlement() {
            return null;
        }

        @SuppressWarnings("all")
        public int getY() {
            return 1;
        }

        @SuppressWarnings("all")
        public int getX() {
            return 0;
        }
    }

    @SuppressWarnings("all")
    private class Settlement {
        @SuppressWarnings("all")
        public Object getOwner() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class ChoiceItem {
        public ChoiceItem(String s, Settlement s1) {
        }
    }

    @SuppressWarnings("all")
    private class NoTestsRemainException extends Exception {
    }

    @SuppressWarnings("all")
    private class Filter {
        @SuppressWarnings("all")
        public boolean shouldRun(Object description) {
            return false;
        }

        @SuppressWarnings("all")
        public void apply(Runner runner) {

        }
    }

    @SuppressWarnings("all")
    private class Runner {
        @SuppressWarnings("all")
        public Object getDescription() {
            return null;
        }
    }

    @SuppressWarnings("all")
    public class MapTransform {
    }

    @SuppressWarnings("all")
    private class FreeColClient {
        @SuppressWarnings("all")
        public FreeColClient getActionManager() {
            return null;
        }

        @SuppressWarnings("all")
        public Object getFreeColAction(Object id) {
            return null;
        }

        @SuppressWarnings("all")
        public FreeColClient getImageLibrary() {
            return null;
        }

        @SuppressWarnings("all")
        public Object getUnitButtonImageIcon(int unitButtonDisband, int i) {
            return null;
        }

        @SuppressWarnings("all")
        public FreeColClient getCanvas() {
            return null;
        }

        @SuppressWarnings("all")
        public void repaint(int i, int i1, Object width, Object height) {

        }

        @SuppressWarnings("all")
        public InGameInputHandler getInGameInputHandler() {
            return null;
        }

        @SuppressWarnings("all")
        public AbstractTranslet getClient() {
            return null;
        }

        @SuppressWarnings("all")
        public Game getGame() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class IterateOverMe {
    }

    @SuppressWarnings("all")
    private class AbstractTranslet {
        @SuppressWarnings("all")
        public void setMessageHandler(InGameInputHandler inGameInputHandler) {

        }
    }

    @SuppressWarnings("all")
    private class CompiledStatement {
        public static final int DELETE = 0;
        public static final int CALL = 1;
        public static final int DDL = 2;
    }



    @SuppressWarnings("all")
    private Object executeCallStatement(Object cs) {
        return null;
    }

    @SuppressWarnings("all")
    private Object executeDDLStatement(Object cs) {
        return null;
    }

    @SuppressWarnings("all")
    private Object executeDeleteStatement(Object cs) {
        return null;
    }

    @SuppressWarnings("all")
    private class ImageLibrary {
        public static final int UNIT_BUTTON_DISBAND = 1;
    }

    @SuppressWarnings("all")
    private void putValue(Object buttonImage, Object unitButtonImageIcon) {

    }

    @SuppressWarnings("all")
    public Tasks_3(FreeColClient freeColClient, String s, Object o, KeyStroke d) {

    }

    @SuppressWarnings("all")
    private static class Reflect {
        @SuppressWarnings("all")
        public static Field resolveJavaField(Class clas, Object name, boolean b) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class Field {
        @SuppressWarnings("all")
        public Object getType() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class LHS {
        @SuppressWarnings("all")
        public LHS(Object object, Field field) {

        }
    }

    @SuppressWarnings("all")
    private class Variable {
        @SuppressWarnings("all")
        public Variable(Object name, Object type, LHS lhs) {
        }
    }

    @SuppressWarnings("all")
    private class AbstractPluginProgramFormating {
        @SuppressWarnings("all")
        public boolean isValid() {
            return false;
        }

        @SuppressWarnings("all")
        public Object getId() {
            return null;
        }

        @SuppressWarnings("all")
        public Object getName() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class ProgramReceiveTarget {
        @SuppressWarnings("all")
        public ProgramReceiveTarget(Tasks_3 tasks, Object name, Object id) {
        }
    }

    @SuppressWarnings("all")
    private static class DEFAULT_CONFIG {
        @SuppressWarnings("all")
        public static Object getName() {
            return null;
        }

        @SuppressWarnings("all")
        public static Object getId() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class Ns {
        @SuppressWarnings("all")
        public Boolean getCatalogName(Boolean aBoolean) {
            return null;
        }

        @SuppressWarnings("all")
        public void insertSys(Boolean[] row) {

        }

        @SuppressWarnings("all")
        public Object getSchemaName(String clsName) {
            return null;
        }

        @SuppressWarnings("all")
        public Boolean getCatalogName(String clsName) {
            return null;
        }

        @SuppressWarnings("all")
        public void checkColumnsMatch(Object mainColArray, Object refTable, Object refColArray) {

        }
    }

    @SuppressWarnings("all")
    private class Element {
        @SuppressWarnings("all")
        public Object getAttribute(String id) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private static class Message {
        @SuppressWarnings("all")
        public static Element getChildElement(Element element, Object xmlElementTagName) {
            return null;
        }

        @SuppressWarnings("all")
        public String getMessageID() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private Unit getGame() {
        return null;
    }

    @SuppressWarnings("all")
    private Ns jj_consume_token(int eq) {
        return null;
    }

    @SuppressWarnings("all")
    private int jj_ntk() {
        return 0;
    }

    @SuppressWarnings("all")
    private class Program {
        @SuppressWarnings("all")
        public void unmark(Object mMarker) {

        }
    }

    @SuppressWarnings("all")
    private class PluginTreeNode {
    }

    @SuppressWarnings("all")
    private PluginTreeNode findProgramTreeNode(Program program, boolean b) {
        return null;
    }

    @SuppressWarnings("all")
    private class IndexRowIterator {
        public Interval next;
        public Tasks_3 last;
    }

    @SuppressWarnings("all")
    private class Interval {
        public IndexRowIterator last;
    }

    @SuppressWarnings("all")
    private static class Messages {
        @SuppressWarnings("all")
        public static String message(String height) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class This {
    }

    @SuppressWarnings("all")
    private class Interpreter {
    }

    @SuppressWarnings("all")
    private This getThis(Interpreter declaringInterpreter) {
        return null;
    }

    @SuppressWarnings("all")
    private class ObjectInputStream {
        @SuppressWarnings("all")
        public int readInt() {
            return 0;
        }

        @SuppressWarnings("all")
        public Object readUTF() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class ResultConstants {
        public static final int SQLDISCONNECT = 0;
    }

    @SuppressWarnings("all")
    private void setOpaque(boolean b) {

    }

    @SuppressWarnings("all")
    private void stopBlinking() {

    }

    @SuppressWarnings("all")
    private boolean hasFocus() {
        return false;
    }

    @SuppressWarnings("all")
    private class Cc {
        @SuppressWarnings("all")
        public String xy(int i, int i1) {
            return null;
        }

        @SuppressWarnings("all")
        public Object msg(String help, String no_endtime_defined) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class TimeDateChooserPanel {
        @SuppressWarnings("all")
        public TimeDateChooserPanel(Object date) {
        }
    }

    @SuppressWarnings("all")
    private class Klass {
        @SuppressWarnings("all")
        public SuiteMethod getMethod(String suite) {
            return null;
        }

        @SuppressWarnings("all")
        public String getName() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class SuiteMethod {
        @SuppressWarnings("all")
        public Object getModifiers() {
            return null;
        }

        @SuppressWarnings("all")
        public Object invoke(Object o) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private static class Modifier {
        @SuppressWarnings("all")
        public static boolean isStatic(Object modifiers) {
            return false;
        }
    }

    @SuppressWarnings("all")
    private class Test {
        //ADDED BY KOBI
        public Class expected() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class Types {
        public static final int VARCHAR = 1;
    }

    @SuppressWarnings("all")
    private void addColumn(Ns t, String procedure_name, int varchar, boolean b) {

    }

    @SuppressWarnings("all")
    private void addColumn(Ns t, String procedure_cat, int varchar) {

    }

    @SuppressWarnings("all")
    private class MissionChip {
        @SuppressWarnings("all")
        public Object get(Object color) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private void loadMissionChip(GraphicsConfiguration gc, Object color, boolean expertMission) {

    }

    @SuppressWarnings("all")
    private class NameSpace {
    }

    @SuppressWarnings("all")
    private static class DriverFactory {
        @SuppressWarnings("all")
        public static DriverFactory getInstance() {
            return null;
        }

        @SuppressWarnings("all")
        public DeviceIf createDevice(String classname, String devname) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class InitializationError extends Throwable {
        @SuppressWarnings("all")
        public InitializationError(String format) {
        }
    }

    @SuppressWarnings("all")
    private static class Request {
        @SuppressWarnings("all")
        public static Request errorReport(Object fTestClass, InitializationError error) {
            return null;
        }

        @SuppressWarnings("all")
        public Object getRunner() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class Session {
        public static final int INFO_CONNECTION_READONLY = 0;
    }

    @SuppressWarnings("all")
    private Object getAttribute(int infoConnectionReadonly) {
        return null;
    }

    @SuppressWarnings("all")
    private class WareHouse {
        @SuppressWarnings("all")
        public boolean getResponseBoolean() {
            return false;
        }
    }

    @SuppressWarnings("all")
    private void remove(WareHouse warehouseDialog) {

    }

    @SuppressWarnings("all")
    private class Cv {
        @SuppressWarnings("all")
        public void visitInsn(Object opcode) {

        }
    }

    @SuppressWarnings("all")
    private Object methodDescription(Method method) {
        return null;
    }

    @SuppressWarnings("all")
    private Object getName() {
        return null;
    }

    @SuppressWarnings("all")
    private class ProgramTable {
        public ProgramTable Plugin;

        @SuppressWarnings("all")
        public void changeSelection(Boolean[] row, int i, boolean b, boolean b1) {

        }

        @SuppressWarnings("all")
        public Object getValueAt(Boolean[] row, int i) {
            return null;
        }

        @SuppressWarnings("all")
        public ProgramTable getPluginManager() {
            return null;
        }

        @SuppressWarnings("all")
        public JPopupMenu createPluginContextMenu(Program p, Object instance) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private static class CapturePlugin {
        @SuppressWarnings("all")
        public static Object getInstance() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class TestResult {
        @SuppressWarnings("all")
        public void addListener(Object adaptingListener) {

        }
    }

    @SuppressWarnings("all")
    private class Ftest {
        @SuppressWarnings("all")
        public void run(TestResult result) {

        }
    }

    @SuppressWarnings("all")
    private Object createAdaptingListener(RunNotifier notifier) {
        return null;
    }

    @SuppressWarnings("all")
    private class Tc {
        public Tc core;
        public Object refTable;
        public Object mainColArray;
        public Object refColArray;
    }

    @SuppressWarnings("all")
    private class TableWorks {
        @SuppressWarnings("all")
        public TableWorks(Connection session, Ns t) {
        }
    }

    @SuppressWarnings("all")
    private class MouseEvent {
        @SuppressWarnings("all")
        public int getY() {
            return 0;
        }

        @SuppressWarnings("all")
        public int getX() {
            return 0;
        }

        @SuppressWarnings("all")
        public Object getSource() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private static class Modifiers {
        @SuppressWarnings("all")
        public boolean hasModifier(String aPublic) {
            return false;
        }
    }

    @SuppressWarnings("all")
    private Object getListCellRendererComponent(JList list, Object value, int index, boolean isSelected, boolean cellHasFocus) {
        return null;
    }

    @SuppressWarnings("all")
    private static class Assert {
        @SuppressWarnings("all")
        public static String format(String message, String fExpected, String fActual) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private void findCommonSuffix() {

    }

    @SuppressWarnings("all")
    private void findCommonPrefix() {

    }

    @SuppressWarnings("all")
    private boolean areStringsEqual() {
        return false;
    }

    @SuppressWarnings("all")
    public static class SwingUtilities {
        @SuppressWarnings("all")
        public static Point convertPoint(Component source, int x, int y, Object o) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private Object iterateRoutineMethods(String className, Object andAliases) {
        return null;
    }

    @SuppressWarnings("all")
    private class Device {
        @SuppressWarnings("all")
        public Object getDriver() {
            return null;
        }

        @SuppressWarnings("all")
        public Object getName() {
            return null;
        }

        @SuppressWarnings("all")
        public void writeData(ObjectOutput out) {

        }
    }

    @SuppressWarnings("all")
    private static class Plugin {
        @SuppressWarnings("all")
        public static Plugin getPluginManager() {
            return null;
        }

        @SuppressWarnings("all")
        public Plugin getTvBrowserSettings() {
            return null;
        }

        @SuppressWarnings("all")
        public File getTvBrowserUserHome() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class Connection {
        public Database database;

        @SuppressWarnings("all")
        public void commit() {

        }
    }

    @SuppressWarnings("all")
    private static class Column {
        @SuppressWarnings("all")
        public static int compare(Object collation, int i, int i1, int coltype) {

            return i;
        }
    }

    @SuppressWarnings("all")
    private void showMainPanel() {

    }

    @SuppressWarnings("all")
    private void removeInGameComponents() {

    }

    @SuppressWarnings("all")
    private void closeMenus() {

    }

    @SuppressWarnings("all")
    private class GUIMessage {
    }

    @SuppressWarnings("all")
    private class Messagee {
        @SuppressWarnings("all")
        public void remove(int i) {

        }

        @SuppressWarnings("all")
        public void add(GUIMessage message) {

        }
    }

    @SuppressWarnings("all")
    private Object getWidth() {
        return null;
    }

    @SuppressWarnings("all")
    private Object getHeight() {
        return null;
    }

    @SuppressWarnings("all")
    private Object getMessageCount() {
        return null;
    }

    @SuppressWarnings("all")
    private int getState() {
        return 0;
    }

    @SuppressWarnings("all")
    private class ViewMode {
        public static final int VIEW_TERRAIN_MODE = 0;
    }

    @SuppressWarnings("all")
    private class Gui {
        @SuppressWarnings("all")
        public Position getSelectedTile() {
            return null;
        }

        @SuppressWarnings("all")
        public TerrainCursor getCursor() {
            return null;
        }

        @SuppressWarnings("all")
        public void setInGame(boolean b) {

        }
    }

    @SuppressWarnings("all")
    private class Position {
        @SuppressWarnings("all")
        public int getY() {
            return 1;
        }

        @SuppressWarnings("all")
        public int getX() {
            return 0;
        }
    }

    @SuppressWarnings("all")
    private class TerrainCursor {
    }

    @SuppressWarnings("all")
    private void jjStopStringLiteralDfa_0(int i, long l, long active1, long active2) {

    }

    @SuppressWarnings("all")
    private int jjStartNfa_0(int i, long l, long old1, long old2) {
        return 0;
    }

    @SuppressWarnings("all")
    private class ElgatoChannel {
    }

    @SuppressWarnings("all")
    private class Script {
        @SuppressWarnings("all")
        public String executeScript(Object channellist) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class Fklass {
        @SuppressWarnings("all")
        public Method[] getMethods() {
            return new Method[0];
        }
    }

    @SuppressWarnings("all")
    private class Annotation {
        @SuppressWarnings("all")
        public Parameters annotationType() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class Parameters {
    }

    @SuppressWarnings("all")
    private class Node {
        @SuppressWarnings("all")
        public Node getRight() {
            return null;
        }

        @SuppressWarnings("all")
        public Node getLeft() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class InGameInputHandler {
    }

    @SuppressWarnings("all")
    private class Game {
        @SuppressWarnings("all")
        public Map getMap() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private Parameters showParametersDialog() {
        return null;
    }

    @SuppressWarnings("all")
    public class AbstractAction {
        @SuppressWarnings("all")
        public void putValue(String name, Object msg) {

        }
    }

    @SuppressWarnings("all")
    private Object createImageIcon(String mimetypes, String s, int i) {
        return null;
    }

    @SuppressWarnings("all")
    private void showDialog() {

    }

    @SuppressWarnings("all")
    private class Count {
        @SuppressWarnings("all")
        public int testCount() {
            return 0;
        }

        @SuppressWarnings("all")
        public Object testAt(int i) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private Object makeDescription(Object testAt) {
        return null;
    }

    @SuppressWarnings("all")
    private class Mexpression {
        public boolean exprType;
    }
}

