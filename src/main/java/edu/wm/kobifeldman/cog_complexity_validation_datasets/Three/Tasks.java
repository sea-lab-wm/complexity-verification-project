package edu.wm.kobifeldman.cog_complexity_validation_datasets.Three;

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
import java.util.logging.Level;

/**
 * Note: method names were changed from 'main' to main + their task number
 *       this allows the code to be compiled and analyzed by sonarqube
 */
public class Tasks {

    private static final String RETURN = "";
    private static final int IRETURN = 0;
    private static final @SuppressWarnings("all") Object BUTTON_IMAGE = null;
    private static final @SuppressWarnings("all") Object BUTTON_ROLLOVER_IMAGE = null;
    private static final int EQ = 0;
    private static final int NE = 1;
    private static final int COLUMNS = 0;
    private static final int DEFAULT_WIDTH = 1;
    private static final int DEFAULT_HEIGHT = 2;
    private static final @SuppressWarnings("all") Locale CONSTRUCTOR_ERROR_FORMAT = null;
    private static final int ACC_PUBLIC = 0;
    private static final @SuppressWarnings("all") Object MESSAGE_COUNT = null;
    private static final int DATABASE_CLOSING = 0;
    private static final int DATABASE_ONLINE = 1;
    private static final @SuppressWarnings("all") Object CHANNELLIST = null;
    private static final boolean COLUMN = false;
    private static final boolean VALUE = false;
    private static final boolean FUNCTION = false;
    private static final boolean ALTERNATIVE = false;
    private static final boolean CASEWHEN = false;
    private static final boolean CONVERT = false;
    private static ScriptEngine body;
    private static String callstack;
    private static Bindings interpreter;
    private static LinkedList<ActionMenu> actionList;
    private static IdentityHashMap<Object, Object> databaseIDMap;
    private static Spring option;
    private static JSpinner spinner;
    private static Object tempCalDefault;
    private static int[] jjbitVec0;
    private static int[] jjbitVec1;
    private int fParameterSetNumber;
    private @SuppressWarnings("all") Object fParameters = null;
    private int constType;
    private ConstraintCore core = new ConstraintCore();
    private HsqlName constName = new HsqlName();
    private String DELTA_START;
    private int fPrefix;
    private int fSuffix;
    private String DELTA_END;
    private boolean windowed;
    private GraphicsDevice gd;
    private Object xsp;
    private Object jj_scanpos;
    private IntValueHashMap rightsMap;
    private String granteeName;
    private GranteeManager granteeManager;
    private Ftest fTest;
    private Method fMethod;
    private Parent parent;
    private InGameController inGameController;
    private Object clas;
    private String value;
    private Object asClass;
    private BtPanel btPanel;
    private Object cancel;
    private Object ok;
    private BtPanel panel;
    private List<Object> nameList;
    private Object sourceFileInfo;
    private int num;
    private Vector<DeviceIf> mDevices;
    private In in;
    private boolean active;
    private int ON;
    private int OFF;
    private Object returnType;
    private Object mCloneData;
    private Object mOwner;
    private CapturePluginPanel mPanel;
    private Object mCurrentPanel;
    private Object player;
    private Settlement inSettlement;
    private BtPanel destinations;
    private int len;
    private int bufpos;
    private Object ret;
    private int bufsize;
    private Object buffer;
    private Vector<Runner> fRunners;
    private OptionalDataException parameters;
    private int outlen;
    private int offset;
    private MapTransform currentMapTransform;
    private FreeColClient freeColClient;
    private Object iterateOverMe;
    private int mYear;
    private int mMonth;
    private int fContextLength;
    private String fExpected;
    private String fActual;
    private Object cs;
    private Object object;
    private Object name;
    private List<AbstractPluginProgramFormating> mConfigs;
    private BtPanel list;
    private Package method;
    private Boolean[] row;
    private Ns ns;
    private Object defschema;
    private Object schema;
    private Ns t;
    private int jj_ntk;
    private ArrayList mChildNodes;
    private Object mMarker;
    private Interval next;
    private int mNumber;
    private Object mName;
    private boolean isClosed;
    private Result resultOut;
    private boolean blinkOn;
    private Cc cc;
    private Cc mLocalizer;
    private Unit UiUtilities;
    private Object date;
    private TimeDateChooserPanel mTimePanel;
    private Klass klass;
    private SuiteMethod suiteMethod;
    private Test suite;
    private Object missionChip;
    private boolean expertMission;
    private Object color;
    private MissionChip expertMissionChips;
    private Stack stack;
    private Class<Object> runnerClass;
    private Object fTestClass;
    private boolean isReadOnly;
    private WareHouse warehouseDialog;
    private Object FRETURN;
    private Object opcode;
    private Object LRETURN;
    private Cv cv;
    private int version;
    private String dataServiceId;
    private List<Method> fTestMethods;
    private Iterator<Object> classNames;
    private String clsName;
    private Boolean clsCat;
    private Object clsSchem;
    private Message[] messages;
    private ProgramTable mProgramTable;
    private ProgramTable mProgramTableModel;
    private ProgramTable devplugin;
    private Tc tc;
    private Connection session;
    private JInternalFrame f;
    private Point loc;
    private ArrayDeque<Object> classNameSet;
    private String className;
    private Object andAliases;
    private Object methods;
    private Device device;
    private ObjectOutput out;
    private Object mCount;
    private Device dev;
    private int fieldcount;
    private int[] cols;
    private int[] a;
    private int[] coltypes;
    private int[] b;
    private Vector importedPackages;
    private String cmpDataServiceId;
    private Tasks cmp;
    private ConstraintCore mainCols;
    private Object refCols;
    private Messagee messages2;
    private Object fMessage;
    private int currentMode;
    private Gui gui;
    private DataInput input_stream;
    private char curChar;
    private Script mAppleScript;
    private Fklass fKlass;
    private Node x;
    private Count ts;
    private Mexpression expression;
    private String ddl;
    private HashMap<Object, Object> messagesToIgnore;

    public Tasks(Class<?> klass) {

    }

    public Tasks(String message) {

    }

    // Snippet s1
    public static Object s1() throws ScriptException {
        Object ret = body.eval(callstack, interpreter);

        boolean breakout = false;
        if(ret instanceof ReturnControl)
        {
            switch(((ReturnControl)ret).kind )
            {
                case RETURN:
                    return ret;
            } // had to be added to allow compilation
        } // had to be added to allow compilation
        return ret; // had to be added to allow compilation
    }

    // Snippet s2
    public static Object s2() {
        if (actionList.size() == 1) {
            ActionMenu menu = actionList.get(0);

            if (menu.getSubItems().length == 0) {
                return null;
            }

            if (menu.getSubItems().length == 1) {
                Action action = menu.getSubItems()[0].getAction();
            } // had to be added to allow compilation
        } // had to be added to allow compilation
        return new Object();                                                /*Altered return*/
        //return null; // had to be added to allow compilation
    }

    // Snippet s2
    public static void s3() {
        if (actionList.size() == 1) {
            ActionMenu menu = actionList.get(0);

            if (menu.getSubItems().length == 0) {
                return;                                                /* ***********************Altered from "return null;"************************** */
            }

            if (menu.getSubItems().length == 1) {
                Action action = menu.getSubItems()[0].getAction();
            } // had to be added to allow compilation
        } // had to be added to allow compilation
    }

    // Snippet s3
    private String compactString(String source) {
        String result = DELTA_START + source.substring(fPrefix, source.length() - fSuffix + 1) + DELTA_END;
        if (fPrefix > 0)
            result = computeCommonPrefix() + result;
        if (fSuffix > 0)
            result = result + computeCommonSuffix();
        return result; // had to be added to allow compilation
    }

    // Snippet s4
    /**
     * Constructor, with a argument reference to the PUBLIC User Object which
     * is null if this is the SYS or PUBLIC user.
     *
     * The dependency upon a GranteeManager is undesirable.  Hopefully we
     * can get rid of this dependency with an IOC or Listener re-design.
     */
    public void Grantee(String name, Grantee inGrantee, // public void added to allow compilation
            GranteeManager man) throws HsqlException {

        rightsMap = new IntValueHashMap();
        granteeName = name;
        granteeManager = man;
    }

    // Snippet s5
    /**
     * Quits the application without any questions.
     */
    public void quit() {
        getConnectController().quitGame(true);
        if (!windowed) {
            gd.setFullScreenWindow(null);
        }
        System.exit(0);
    }   // had to be added to allow compilation

    // Snippet s6
    private boolean s6() {
        xsp = jj_scanpos;
        if (jj_scan_token(100)) {
            jj_scanpos = xsp;
            if (jj_scan_token(101)) return true;
        } // had to be added to allow compilation
        return true; // had to be added to allow compilation
    }

    // Snippet s7
    /**
     * Attention: DO NOT USE THIS!
     * Under Os/2 it has some problems with calculating the real Date!
     *
     * @deprecated
     */
    public void Date(int daysSince1970) { // return type void added to allow compilation

        long l = (long) daysSince1970 * 24 * 60 * 60 * 1000;
        java.util.Date d = new java.util.Date(l);
        Calendar cal = Calendar.getInstance();
    } // added to allow compilation

    // Snippet s8
    public void TestMethodRunner(Object test, Method method, RunNotifier notifier, Description description) {
        super1(test.getClass(), Before.class, After.class, test); // super() renamed to super1() to allow compilation
        fTest= (Ftest) test; // Type cast to Ftest to allow compilation
        fMethod= method;
    } // added to allow compilation

    // Snippet s9
    /**
     * Returns a vector containing the URI (type + path) for all the databases.
     */
    public static Vector getDatabaseURIs() {

        Vector v = new Vector();
        Iterator it = databaseIDMap.values().iterator();

        while (it.hasNext()) {
            Database db = (Database) it.next();
        } // added to allow compilation
        return v;                                                       /*Altered return*/
        //return null; // added to allow compilation
    } // added to allow compilation

    // Snippet s10
    private void moveUnit(KeyEvent e) {
        if (!parent.isMapboardActionsEnabled()) {
            return;
        }

        switch (e.getKeyCode()) {
            case KeyEvent.VK_ESCAPE:
                // main menu
                break;
            case KeyEvent.VK_NUMPAD1:
            case KeyEvent.VK_END:
                inGameController.moveActiveUnit(Map.SW);
        }
    }

    // Snippet s11
    private Object s11() throws ClassNotFoundException {
        if (clas == null)
            throw new ClassNotFoundException(
                    "Class: " + value + " not found in namespace");

        asClass = clas;
        return asClass;
    }

    // Snippet s12
    private void s12() {
        btPanel.add(cancel);

        getRootPane().setDefaultButton(ok);

        panel.add(btPanel, BorderLayout.SOUTH);
    }

    // Snippet s13
    /*
     * @param expected expected value
	 * @param actual actual value
	 */
    static public void assertEquals(String message, Object expected, Object actual) {
        if (expected == null && actual == null)
            return;
        if (expected != null && isEquals(expected, actual))
            return;
        else if (expected instanceof String && actual instanceof String) {
            String cleanMessage = message == null ? "" : message;
        } // added to allow compilation
    } // added to allow compilation

    // Snippet s14
    Object removeName (String name) throws HsqlException {

        Object owner = nameList.remove(name);

        if (owner == null) {

            // should contain name
            throw Trace.error(Trace.GENERAL_ERROR);
        }

        return owner;
    }// added to allow compilation

    // Snippet s15
    public static void s15() {
        int stepSize = Math.min((option.getMaximumValue() - option.getMinimumValue()) / 10, 1000);
        spinner = new JSpinner(new SpinnerNumberModel(option.getValue(), option.getMinimumValue(),
                option.getMaximumValue(), Math.max(1, stepSize)));
        spinner.setToolTipText(option.toString()); // rename getShortDescription to toString to allow compilation
    }

    // Snippet s16
    public void s16() {
        if ( parent != null )
            setStrictJava( parent.getStrictJava() );
        this.sourceFileInfo = sourceFileInfo;
        BshClassManager bcm = BshClassManager.createClassManager( this );
    }

    // Snippet s17
    public void s17() {
        mDevices = new Vector<DeviceIf>();

        DeviceFileHandling reader = new DeviceFileHandling();

        for (int i = 0; i < num; i++) {
            String classname = (String) in.readObject();
        }
    } // Added to allow compilation

    // Snippet s18
    /**
     * Do not use. Testing purposes only.
     */
    public Result runMain(String... args) {
        System.out.println("JUnit version " + Version.id());
        List<Class<?>> classes = new ArrayList<Class<?>>();
        List<Failure> missingClasses = new ArrayList<Failure>();
        return new Result();                                                        /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s19
    /**
     * temp constraint constructor
     */
    void Constraint(HsqlName name, int[] mainCols, Table refTable, int[] refCols, // Added return type void to allow compilation
               int type, int deleteAction, int updateAction) {

        core              = new ConstraintCore();
        constName         = name;
        constType         = type;
    } // Added to allow compilation

    // Snippet s20
    public void s20() {
        int eventId = active? ON : OFF;
        ActionEvent blinkEvent = new ActionEvent(this,eventId,"blink");

        fireActionEvent(blinkEvent);
    }

    // Snippet s21
    public void s21() {
        if(true) // Added to allow compilation
            System.out.println(""); // Added to allow compilation
		else if ( isPrimitive( returnType ) ) {
            int opcode = IRETURN;
            String type;
            String meth;
        } // Added to allow compilation
    }

    // Snippet s22
    /**
     * Returns the PluginPanel
     * @return Panel
     */
    public JPanel createSettingsPanel() {
        mPanel = new CapturePluginPanel(mOwner, mCloneData);
        mPanel.setBorder(Borders.createEmptyBorder(Sizes.DLUY5, Sizes.DLUX5, Sizes.DLUY5, Sizes.DLUX5));
        mPanel.setSelectedTab(mCurrentPanel);
        return new JPanel();                                                                                    /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s23
    protected void printFailures(Result result) {
        if (result.getFailureCount() == 0)
            return;
        if (result.getFailureCount() == 1)
            getWriter().println("There was " + result.getFailureCount() + " failure:");
        else
            getWriter().println("There were " + result.getFailureCount() + " failures:");
    } // Added to allow compilation

    // Snippet s24
    public static long getNormalisedTime(long t) {

        synchronized (tempCalDefault) {
            setTimeInMillis(tempCalDefault, t);
            resetToTime(tempCalDefault);

            return getTimeInMillis(tempCalDefault);
        }  // Added to allow compilation
    } // Added to allow compilation

    // Snippet s25
    public boolean check(Unit u, PathNode p) {
        if (p.getTile().getSettlement() != null && p.getTile().getSettlement().getOwner() == player
                && p.getTile().getSettlement() != inSettlement) {
            Settlement s = p.getTile().getSettlement();
            int turns = p.getTurns();
            destinations.add(new ChoiceItem(s.toString() + " (" + turns + ")", s));
        }  // Added to allow compilation
        return false; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s26
    public void s26() {
        if ((bufpos + 1) >= len)
            System.arraycopy(buffer, bufpos - len + 1, ret, 0, len);
        else {
            System.arraycopy(buffer, bufsize - (len - bufpos - 1), ret, 0,
                    len - bufpos - 1);
            System.arraycopy(buffer, 0, ret, len - bufpos - 1, bufpos + 1);
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s27
    /**
     * Compute the proper position for a centered window
     */
    private Point comuteDisplayPointCentre(Dimension dim) {
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (screen.width - dim.width) / 2;
        int y = (screen.height - dim.height) / 2;
        return new Point(x, y);                                                     /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s28
    public void filter(Filter filter) throws NoTestsRemainException {
        for (Iterator<Runner> iter= fRunners.iterator(); iter.hasNext();) {
            Runner runner = iter.next();
            if (filter.shouldRun(runner.getDescription()))
                filter.apply(runner);
            else
                iter.remove();
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s29
    public void s29() {
        boolean    hasReturnValue;

        outlen = parameters.length;
        offset = 0;
    }

    // Snippet s30
    /**
     * Sets the currently chosen <code>MapTransform</code>.
     * @param mt The transform that should be applied to a
     *      <code>Tile</code> that is clicked on the map.
     */
    public void setMapTransform(MapTransform mt) {
        currentMapTransform = mt;
        MapControlsAction mca = (MapControlsAction) freeColClient.getActionManager().getFreeColAction(MapControlsAction.ID);
        if (mca.getMapControls() != null) {
            mca.getMapControls().update(mt);
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s31
    public String s31() {
        if (iterateOverMe instanceof String)
            return createEnumeration(((String) iterateOverMe).toCharArray());

        if (iterateOverMe instanceof StringBuffer)
            return createEnumeration(
                    iterateOverMe.toString().toCharArray());

        throw new IllegalArgumentException(
                "Cannot enumerate object of type " + iterateOverMe.getClass());
    }

    private String createEnumeration(char[] toCharArray) {
        return new String();                                                            /*Altered return*/
        //return null;
    }

    // Snippet s32
    /**
     * Constructs a new Date object, initialized with the current date.
     */
    public void Date() { // Return type void added to allow compilation
        Calendar mCalendar = Calendar.getInstance();
        mYear = mCalendar.get(Calendar.YEAR);
        mMonth = mCalendar.get(Calendar.MONTH) + 1;
    } // Added to allow compilation

    // Snippet s33
    /**
     * @param contextLength the maximum length for <code>expected</code> and <code>actual</code>. When contextLength
     * is exceeded, the Strings are shortened
     * @param expected the expected string value
     * @param actual the actual string value
     */
    public void ComparisonCompactor(int contextLength, String expected, String actual) { // return type void added to allow compilation
        fContextLength = contextLength;
        fExpected = expected;
        fActual = actual;
    } // Added to allow compilation

    // Snippet s34
    public Object s34() {
        int statement = 0; // added to allow compilation
        switch (statement) { // Added switch case beginning to allow compilation
            case CompiledStatement.DELETE:
                return executeDeleteStatement(cs);

            case CompiledStatement.CALL:
                return executeCallStatement(cs);

            case CompiledStatement.DDL:
                return executeDDLStatement(cs);
        } // added to allow compilation
        return new Object();                                                                    /*Altered return*/
        //return null; // added return statement to allow compilation
    }

    // Snippet s35
    public class DisbandUnitAction extends Tasks{ // Wrapped in a class to allow compilation
        /**
         * Creates a new <code>DisbandUnitAction</code>.
         *
         * @param freeColClient The main controller object for the client.
         */
        DisbandUnitAction(FreeColClient freeColClient) {
            super(freeColClient, "unit.state.8", null, KeyStroke.getKeyStroke('D', 0));
            putValue(BUTTON_IMAGE, freeColClient.getImageLibrary().getUnitButtonImageIcon(ImageLibrary.UNIT_BUTTON_DISBAND,
                    0));
            putValue(BUTTON_ROLLOVER_IMAGE, freeColClient.getImageLibrary().getUnitButtonImageIcon(
                    ImageLibrary.UNIT_BUTTON_DISBAND, 1));
        }
    }

    // Snippet s36
    public Object s36() {
        Class clas = object.getClass();
        Field field = Reflect.resolveJavaField(
                clas, name, false/*onlyStatic*/);
        if (field != null)
            return new Variable(
                    name, field.getType(), new LHS(object, field));
        return object;                                                                    /*Altered return*/
        //return null; // Added to allow compilation
    }

    // Snippet s37
    public Object s37() {
        for(AbstractPluginProgramFormating config : mConfigs)
            if(config != null && config.isValid())
                list.add(new ProgramReceiveTarget(this, config.getName(), config.getId()));

        if(list.isEmpty())
            list.add(new ProgramReceiveTarget(this, DEFAULT_CONFIG.getName(), DEFAULT_CONFIG.getId()));

        return list.toArray(new ProgramReceiveTarget[list.size()]);
    }

    // Snippet s38
//    public Object s38() {
//        Class<? extends Throwable> expectedException(Method method){
//            Test annotation = method.getAnnotation(Test.class);
//            if (annotation.expected() == None.class)
//                return null;
//            else
//                return annotation.expected();
//        }
//    }

    // Snippet s39
    public void s39() {
        row[1] = ns.getCatalogName(row[0]);
        row[2] = schema.equals(defschema) ? Boolean.TRUE
                : Boolean.FALSE;

        t.insertSys(row);
    } // Added to allow compilation

    // Snippet s40
    /**
     * Handles an "deliverGift"-request.
     *
     * @param element The element (root element in a DOM-parsed XML tree) that
     *            holds all the information.
     */
    private Element deliverGift(Element element) {
        Element unitElement = Message.getChildElement(element, Unit.getXMLElementTagName());

        Unit unit = (Unit) getGame().getFreeColGameObject(unitElement.getAttribute("ID"));
        unit.readFromXMLElement(unitElement);
        return unitElement;                                                                                 /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s41
    public void s41() {
        switch ((jj_ntk==-1)?jj_ntk():jj_ntk) {
            case EQ:
                t = jj_consume_token(EQ);
                break;
            case NE:
                t = jj_consume_token(NE);
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s42
    public synchronized void removeProgram(Program program) {
        PluginTreeNode node = findProgramTreeNode(program, false);
        if (node != null) {
            mChildNodes.remove(node);
            if (mMarker != null) {
                program.unmark(mMarker);
            } // Added to allow compilation
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s43
    public class TestClassRunnerForParameters extends Tasks{ // Added class wrapper to allow compilation
        private TestClassRunnerForParameters(Class<?> klass, Object[] parameters, int i) {
            super(klass);
            fParameters= parameters;
            fParameterSetNumber= i;
        } // Added to allow compilation
    }

    // Snippet s44
    void link(IndexRowIterator other) {

        other.next = next;
        other.last = this;
        next.last  = other;
    } // Added to allow compilation

    // Snippet s45
    public void s45() {
        final String heightText = Messages.message("height");

        final JTextField inputWidth = new JTextField(Integer.toString(DEFAULT_WIDTH), COLUMNS);
        final JTextField inputHeight = new JTextField(Integer.toString(DEFAULT_HEIGHT), COLUMNS);
    } // Added to allow compilation

    // Snippet s46
    /**
     Get the top level namespace or this namespace if we are the top.
     Note: this method should probably return type bsh.This to be consistent
     with getThis();
     */
    public This getGlobal( Interpreter declaringInterpreter )
    {
        if ( parent != null )
            return parent.getGlobal( declaringInterpreter );
        else
            return getThis( declaringInterpreter );
    } // Added to allow compilation

    // Snippet s47
    /**
     * Read Settings
     * @param stream
     * @throws IOException
     * @throws ClassNotFoundException
     */
    public void readData(ObjectInputStream stream) throws IOException, ClassNotFoundException {
        int version = stream.readInt();
        mNumber = stream.readInt();
        mName = stream.readUTF();
    } // Added to allow compilation

    // Snippet s48
    public class ComparisonFailure extends Tasks{ // Class wrapper to allow compilation
        /**
         * Constructs a comparison failure.
         * @param message the identifying message or null
         * @param expected the expected string value
         * @param actual the actual string value
         */
        public ComparisonFailure (String message, String expected, String actual) {
            super (message);
            fExpected= expected;
            fActual= actual;
        } // Added to allow compilation
    }

    // Snippet s49
    public void close() {

        if (isClosed) {
            return;
        }

        isClosed = true;

        try {
            resultOut.setResultType(ResultConstants.SQLDISCONNECT);

        } finally {
            // Added to allow compilation
        }
    } // Added to allow compilation

    // Snippet s50
    public void actionPerformed(ActionEvent evt) {
        if (!hasFocus()) {
            stopBlinking();
        }

        if (blinkOn) {
            setOpaque(false);
            blinkOn = false;
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s51
    private static String getBaseName( String className )
    {
        int i = className.indexOf("$");
        if ( i == -1 )
            return className;

        return className.substring(i+1);
    } // Added to allow compilation

    // Snippet s52
    public void s52() {

        panel.add(UiUtilities.createHelpTextArea(mLocalizer.msg("help","No endtime defined")), cc.xy(1,1));

        mTimePanel = new TimeDateChooserPanel(date);
        panel.add(mTimePanel, cc.xy(1,3));

    } // Added to allow compilation

    // Snippet s53
    public void s53() throws Exception{
        try {
            suiteMethod= klass.getMethod("suite");
            if (! Modifier.isStatic(suiteMethod.getModifiers())) {
                throw new Exception(klass.getName() + ".suite() must be static");
            }
            suite= (Test) suiteMethod.invoke(null); // static method

        } finally {
            // Added to allow compilation
        }
    } // Added to allow compilation

    // Snippet s54
    public void s54() {
        // ----------------------------------------------------------------
        // required
        // ----------------------------------------------------------------
        addColumn(t, "PROCEDURE_CAT", Types.VARCHAR);
        addColumn(t, "PROCEDURE_SCHEM", Types.VARCHAR);
        addColumn(t, "PROCEDURE_NAME", Types.VARCHAR, false);    // not null
    } // Added to allow compilation

    // Snippet s55
    public void s55() {
        if (missionChip == null) {
            GraphicsConfiguration gc = GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice()
                    .getDefaultConfiguration();
            loadMissionChip(gc, color, expertMission);

            if (expertMission) {
                missionChip = expertMissionChips.get(color);
            } // Added to allow compilation
        } // Added to allow compilation
    } // Added to allow compilation


    // Snippet s56
    /**
     Swap in the value as the new top of the stack and return the old
     value.
     */
    public NameSpace swap( NameSpace newTop ) {
        NameSpace oldTop = (NameSpace)(stack.elementAt(0));
        stack.setElementAt( newTop, 0 );
        return oldTop;
    } // Added to allow compilation

    // Snippet s57
    public void s57() {
        String classname = (String) in.readObject();
        String devname = (String)in.readObject();

        DeviceIf dev = DriverFactory.getInstance().createDevice(classname, devname);
    } // Added to allow compilation

    // Snippet s58
    public Object s58() {
        String simpleName= runnerClass.getSimpleName();
        InitializationError error= new InitializationError(String.format(
                CONSTRUCTOR_ERROR_FORMAT, simpleName, simpleName));
        return Request.errorReport(fTestClass, error).getRunner();
    } // Added to allow compilation

    // Snippet s59
    public boolean isReadOnly() throws HsqlException {

        Object info = getAttribute(Session.INFO_CONNECTION_READONLY);

        isReadOnly = ((Boolean) info).booleanValue();

        return isReadOnly;
    } // Added to allow compilation

    // Snippet s60
    public Object s60() {
        boolean response = warehouseDialog.getResponseBoolean();
        remove(warehouseDialog);
        return response;
    } // Added to allow compilation

    // Snippet s61
    public void s61() {
		if (true) // Added to allow compilation
            System.out.println(""); // Added to allow compilation
        else if ( returnType.equals("F") )
            opcode = FRETURN;
        else if ( returnType.equals("J") )  //long
            opcode = LRETURN;

        cv.visitInsn(opcode);
    } // Added to allow compilation

    // Snippet s62
    public void s62() {
        String channelId;

        if (version==1) {
            dataServiceId = (String) in.readObject();
            channelId = "" + in.readInt();
        }
    } // Added to allow compilation

    // Snippet s63
    // @Override // Removed to allow compilation
    public Description getDescription() {
        Description spec = Description.createSuiteDescription(getName());
        List<Method> testMethods = fTestMethods;
        for (Method method : testMethods)
            spec.addChild(methodDescription(method));

        return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s64
    public void s64() {
        while (classNames.hasNext()) {
            clsName = (String) classNames.next();
            clsCat = ns.getCatalogName(clsName);
            clsSchem = ns.getSchemaName(clsName);
        }
    } // Added to allow compilation

    // Snippet s65
    public void s65() {
        String[] texts = new String[messages.length];
        ImageIcon[] images = new ImageIcon[messages.length];
        for (int i = 0; i < messages.length; i++) {
            String ID = messages[i].getMessageID();
        }
    } // Added to allow compilation

    // Snippet s66
    public void s66() {
        mProgramTable.changeSelection(row, 0, false, false);

        Program p = (Program) mProgramTableModel.getValueAt(row, 1);

        JPopupMenu menu = devplugin.Plugin.getPluginManager().createPluginContextMenu(p, CapturePlugin.getInstance());

    } // Added to allow compilation

    // Snippet s67
    /**
     * Add one zero if neccessary
     * @param number
     * @return
     */
    private CharSequence addZero(int number) {
        StringBuilder builder = new StringBuilder();

        if (number < 10) {
            builder.append('0');
        }

        builder.append(Integer.toString(number));
        return builder;                                                                            /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s68
    // @Override // Removed to allow compilation
    public void run(RunNotifier notifier) {
        TestResult result= new TestResult();
        result.addListener(createAdaptingListener(notifier));
        fTest.run(result);
    } // Added to allow compilation


    // Snippet s69
    public void s69() throws SQLException {
        t.checkColumnsMatch(tc.core.mainColArray, tc.core.refTable,
                tc.core.refColArray);
        session.commit();

        TableWorks tableWorks = new TableWorks(session, t);
    } // Added to allow compilation

    // Snippet s70
//    @Override // Removed to allow compilation
    public void mousePressed(MouseEvent e) {
        if (f.getDesktopPane() == null || f.getDesktopPane().getDesktopManager() == null) {
            return;
        }
        loc = SwingUtilities.convertPoint((Component) e.getSource(), e.getX(), e.getY(), null);
        f.getDesktopPane().getDesktopManager().beginDraggingFrame(f);
    } // Added to allow compilation

    // Snippet s71
    /**
     Translate bsh.Modifiers into ASM modifier bitflags.
     */
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
    public class Component extends Tasks{ // Added class wrapper to allow compilation
        public Component(String message) {
            super(message);
        }

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
    public String compact(String message) {
        if (fExpected == null || fActual == null || areStringsEqual())
            return Assert.format(message, fExpected, fActual);

        findCommonPrefix();
        findCommonSuffix();

        return message;                                                                 /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s74
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
    private void loadColorChip(GraphicsConfiguration gc, Color c) {
        BufferedImage tempImage = gc.createCompatibleImage(11, 17);
        Graphics g = tempImage.getGraphics();
        if (c.equals(Color.BLACK)) {
            g.setColor(Color.WHITE);
        }
    } // Added to allow compilation

    // Snippet s76
    public void s76() throws IOException{
        out.writeObject(device.getDriver().getClass().getName());
        out.writeObject(device.getName());

        device.writeData(out);
    } // Added to allow compilation

    // Snippet s77
    public void s77() throws FileNotFoundException, IOException{
        File data = new File(Plugin.getPluginManager().getTvBrowserSettings().getTvBrowserUserHome()  + File.separator +
                "CaptureDevices" + File.separator + mCount + ".dat");

        ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(data));

        dev.writeData(stream);
    } // Added to allow compilation

    // Snippet s78
//    private static Class<?>[] getAnnotatedClasses(Class<?> klass) throws InitializationError {
//            SuiteClasses annotation= klass.getAnnotation(SuiteClasses.class);
//            if (annotation == null)
//                throw new InitializationError(String.format("class '%s' must have a SuiteClasses annotation", klass.getName()));
//            return annotation.value();
//    } // Added to allow compilation

    // Snippet s79
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
    public synchronized void addMessage(GUIMessage message) {
        if (getMessageCount() == MESSAGE_COUNT) {
            messages2.remove(0); // Renamed to allow compilation
        }
        messages2.add(message); // Renamed to allow compilation

        freeColClient.getCanvas().repaint(0, 0, getWidth(), getHeight());
    } // Added to allow compilation

    // Snippet s86
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
    public String getMessage() {
        StringBuilder builder= new StringBuilder();
        if (fMessage != null)
            builder.append(fMessage);
        builder.append("arrays first differed at element ");

        return new String();                                                            /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s89
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

    // Snippet s93
//    private Method getParametersMethod() throws Exception {
//        for (Method each : fKlass.getMethods()) {
//            if (Modifier.isStatic(each.getModifiers())) {
//                Annotation[] annotations= each.getAnnotations();
//                for (Annotation annotation : annotations) {
//                    if (annotation.annotationType() == Parameters.class)
//                        return each;
//                }
//            }
//        }
//        throw new Exception("No public static parameters method on class "
//                + getName());
//    } // Added to allow compilation

    // Snippet s94
    public void s94() {
        Node r = x.getRight();

        if (r != null) {
            x = r;

            Node l = x.getLeft();
        }
    } // Added to allow compilation

    // Snippet s95
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
    public void actionPerformed2(ActionEvent e) { // Renamed to allow compilation
        final Game game = freeColClient.getGame();
        final Map map = game.getMap();

        Parameters p = showParametersDialog();
    } // Added to allow compilation

    // Snippet s97
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
    public void s98() {
        Description description= Description.createSuiteDescription(name);
        int n= ts.testCount();
        for (int i= 0; i < n; i++)
            description.addChild(makeDescription(ts.testAt(i)));
    } // Added to allow compilation

    // Snippet s99
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

    // Snippet s100
//    private synchronized void purgeOldMessagesFromMessagesToIgnore(int thisTurn) {
//        List<String> keysToRemove = new ArrayList<String>();
//        for (Entry<String, Integer> entry : messagesToIgnore.entrySet()) {
//            if (entry.getValue().intValue() < thisTurn - 1) {
//                if (logger.isLoggable(Level.FINER)) {
//                    logger.finer("Removing old model message with key " + entry.getKey() + " from ignored messages.");
//                }
//                keysToRemove.add(entry.getKey());
//            }
//        }
//    } // Added to allow compilation

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

    private static class ActionMenu {
        @SuppressWarnings("all")
        public AbstractButton[] getSubItems() {
            return new AbstractButton[0];
        }
    }

    private static class GranteeManager {
    }

    public static class HsqlException extends Exception {
    }

    private static class Database {
        public Object collation;
    }

    private class ConnectController {
        @SuppressWarnings("all")
        public void quitGame(boolean b) {

        }
    }

    private class Grantee {
    }

    private class IntValueHashMap {
    }

    private class Method {
        @SuppressWarnings("all")
        public Object getModifiers() {
            return null;
        }

        @SuppressWarnings("all")
        public Annotation[] getAnnotations() {
            return new Annotation[0];
        }
    }

    private class RunNotifier {
    }

    private static class Description {
        @SuppressWarnings("all")
        public static Description createSuiteDescription(Object name) {
            return null;
        }

        @SuppressWarnings("all")
        public void addChild(Object methodDescription) {

        }
    }

    private class Before {
    }

    private class After {
    }

    private static class BshClassManager {
        @SuppressWarnings("all")
        public static BshClassManager createClassManager(Tasks tasks) {
            return null;
        }
    }

    private class DeviceIf {
    }

    private class DeviceFileHandling {
    }

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

    private static class Version {
        @SuppressWarnings("all")
        public static String id() {
            return null;
        }
    }

    private class Failure {
    }

    private class HsqlName {
    }

    private class Table {
    }

    private class ConstraintCore {
        public ConstraintCore mainColArray;
        public Object colLen;
        public Object length;
        public Object refColArray;
    }

    private class ActionEvent {
        @SuppressWarnings("all")
        public ActionEvent(Tasks tasks, int eventId, String blink) {
        }
    }

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

    private class Settlement {
        @SuppressWarnings("all")
        public Object getOwner() {
            return null;
        }
    }

    private class ChoiceItem {
        public ChoiceItem(String s, Settlement s1) {
        }
    }

    private class NoTestsRemainException extends Exception {
    }

    private class Filter {
        @SuppressWarnings("all")
        public boolean shouldRun(Object description) {
            return false;
        }

        @SuppressWarnings("all")
        public void apply(Runner runner) {

        }
    }

    private class Runner {
        @SuppressWarnings("all")
        public Object getDescription() {
            return null;
        }
    }

    public class MapTransform {
    }

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

    private class IterateOverMe {
    }

    private class AbstractTranslet {
        @SuppressWarnings("all")
        public void setMessageHandler(InGameInputHandler inGameInputHandler) {

        }
    }

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

    private class ImageLibrary {
        public static final int UNIT_BUTTON_DISBAND = 1;
    }

    @SuppressWarnings("all")
    private void putValue(Object buttonImage, Object unitButtonImageIcon) {

    }

    @SuppressWarnings("all")
    public Tasks(FreeColClient freeColClient, String s, Object o, KeyStroke d) {

    }

    private static class Reflect {
        @SuppressWarnings("all")
        public static Field resolveJavaField(Class clas, Object name, boolean b) {
            return null;
        }
    }

    private class Field {
        @SuppressWarnings("all")
        public Object getType() {
            return null;
        }
    }

    private class LHS {
        @SuppressWarnings("all")
        public LHS(Object object, Field field) {

        }
    }

    private class Variable {
        @SuppressWarnings("all")
        public Variable(Object name, Object type, LHS lhs) {
        }
    }

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

    private class ProgramReceiveTarget {
        @SuppressWarnings("all")
        public ProgramReceiveTarget(Tasks tasks, Object name, Object id) {
        }
    }

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

    private class Element {
        @SuppressWarnings("all")
        public Object getAttribute(String id) {
            return null;
        }
    }

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

    private class Program {
        @SuppressWarnings("all")
        public void unmark(Object mMarker) {

        }
    }

    private class PluginTreeNode {
    }

    @SuppressWarnings("all")
    private PluginTreeNode findProgramTreeNode(Program program, boolean b) {
        return null;
    }

    private class IndexRowIterator {
        public Interval next;
        public Tasks last;
    }

    private class Interval {
        public IndexRowIterator last;
    }

    private static class Messages {
        @SuppressWarnings("all")
        public static String message(String height) {
            return null;
        }
    }

    private class This {
    }

    private class Interpreter {
    }

    @SuppressWarnings("all")
    private This getThis(Interpreter declaringInterpreter) {
        return null;
    }

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

    private class TimeDateChooserPanel {
        @SuppressWarnings("all")
        public TimeDateChooserPanel(Object date) {
        }
    }

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

    private static class Modifier {
        @SuppressWarnings("all")
        public static boolean isStatic(Object modifiers) {
            return false;
        }
    }

    private class Test {
    }

    private class Types {
        public static final int VARCHAR = 1;
    }

    @SuppressWarnings("all")
    private void addColumn(Ns t, String procedure_name, int varchar, boolean b) {

    }

    @SuppressWarnings("all")
    private void addColumn(Ns t, String procedure_cat, int varchar) {

    }

    private class MissionChip {
        @SuppressWarnings("all")
        public Object get(Object color) {
            return null;
        }
    }

    @SuppressWarnings("all")
    private void loadMissionChip(GraphicsConfiguration gc, Object color, boolean expertMission) {

    }

    private class NameSpace {
    }

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

    private class InitializationError {
        @SuppressWarnings("all")
        public InitializationError(String format) {
        }
    }

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

    private class Session {
        public static final int INFO_CONNECTION_READONLY = 0;
    }

    @SuppressWarnings("all")
    private Object getAttribute(int infoConnectionReadonly) {
        return null;
    }

    private class WareHouse {
        @SuppressWarnings("all")
        public boolean getResponseBoolean() {
            return false;
        }
    }

    @SuppressWarnings("all")
    private void remove(WareHouse warehouseDialog) {

    }

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

    private static class CapturePlugin {
        @SuppressWarnings("all")
        public static Object getInstance() {
            return null;
        }
    }

    private class TestResult {
        @SuppressWarnings("all")
        public void addListener(Object adaptingListener) {

        }
    }

    private class Ftest {
        @SuppressWarnings("all")
        public void run(TestResult result) {

        }
    }

    @SuppressWarnings("all")
    private Object createAdaptingListener(RunNotifier notifier) {
        return null;
    }

    private class Tc {
        public Tc core;
        public Object refTable;
        public Object mainColArray;
        public Object refColArray;
    }

    private class TableWorks {
        @SuppressWarnings("all")
        public TableWorks(Connection session, Ns t) {
        }
    }

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

    private class Connection {
        public Database database;

        @SuppressWarnings("all")
        public void commit() {

        }
    }

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

    private class GUIMessage {
    }

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

    private class ViewMode {
        public static final int VIEW_TERRAIN_MODE = 0;
    }

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

    private class TerrainCursor {
    }

    @SuppressWarnings("all")
    private void jjStopStringLiteralDfa_0(int i, long l, long active1, long active2) {

    }

    @SuppressWarnings("all")
    private int jjStartNfa_0(int i, long l, long old1, long old2) {
        return 0;
    }

    private class ElgatoChannel {
    }

    private class Script {
        @SuppressWarnings("all")
        public String executeScript(Object channellist) {
            return null;
        }
    }

    private class Fklass {
        @SuppressWarnings("all")
        public Method[] getMethods() {
            return new Method[0];
        }
    }

    private class Annotation {
        @SuppressWarnings("all")
        public Parameters annotationType() {
            return null;
        }
    }

    private class Parameters {
    }

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

    private class InGameInputHandler {
    }

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

    public class AbstractAction {
        @SuppressWarnings("all")
        public void putValue(String name, Object msg) {

        }
    }

    @SuppressWarnings("all")
    private Object createImageIcon(String mimetypes, String s, int i) {
        return null;
    }

    private void showDialog() {

    }

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

    private class Mexpression {
        public boolean exprType;
    }
}

