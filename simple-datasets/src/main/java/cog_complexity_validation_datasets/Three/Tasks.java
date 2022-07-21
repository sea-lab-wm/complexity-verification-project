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
public class Tasks {

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
    private Tasks cmp;
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
    public Tasks(Class<?> klass) {

    }

    @SuppressWarnings("all")
    public Tasks(String message) {

    }

    //ADDED BY KOBI
    @SuppressWarnings("all")
    public void runAllSnippets() {
        //S1
        try {
            s1();
        } catch (ScriptException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S2
        s2();
        //S3
        compactString("source");
        //S4
        try {
            Grantee("name", new Grantee(), new GranteeManager());
        } catch (HsqlException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S5
        quit();
        //S6
        s6();
        //S7
        Date(1000);
        //S8
        TestMethodRunner(new Object(), new Method(), new RunNotifier(), new Description());
        //S9
        getDatabaseURIs();
        //S10
        moveUnit(new KeyEvent(f, 1, 1, 1, 1, 'c'));
        //S11
        try {
            s11();
        } catch (ClassNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S12
        s12();
        //S13
        assertEquals("message", new Object(), new Object());
        //S14
        try {
            removeName("name");
        } catch (HsqlException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S15
        s15();
        //S16
        s16();
        //S17
        s17();
        //S18
        runMain("args");
        //S19
        Constraint(new HsqlName(), new int[10], new Table(), new int[10], 1, 1, 1);
        //S20
        s20();
        //S21
        s21();
        //S22
        createSettingsPanel();
        //S23
        printFailures(new Result());
        //S24
        getNormalisedTime(1);
        //S25
        check(new Unit(), new PathNode());
        //S26
        s26();
        //S27
        comuteDisplayPointCentre(new Dimension());
        //S28
        try {
            filter(new Filter());
        } catch (NoTestsRemainException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S29
        s29();
        //S30
        setMapTransform(new MapTransform());
        //S31
        s31();
        //S32
        Date();
        //S33
        ComparisonCompactor(1, "expected", "actual");
        //S34
        s34();
        //S35
        new DisbandUnitAction(new FreeColClient());
        //S36
        s36();
        //S37
        s37();
        //S38
        //SKIPPED!!!!!!!!!!!!!!!
        //S39
        s39();
        //S40
        deliverGift(new Element());
        //S41
        s41();
        //S42
        removeProgram(new Program());
        //S43
        new TestClassRunnerForParameters(new Klass().getClass(), new Object[10], 1);
        //S44
        link(new IndexRowIterator());
        //S45
        s45();
        //S46
        getGlobal(new Interpreter());
        //S47
        try {
            readData(new ObjectInputStream());
        } catch (ClassNotFoundException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        } catch (IOException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S48
        new ComparisonFailure("message", "expected", "actual");
        //S49
        close();
        //S50
        actionPerformed(new ActionEvent(new Tasks("message"), 1, "blink"));
        //S51
        getBaseName("className");
        //S52
        s52();
        //S53
        try {
            s53();
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S54
        s54();
        //S55
        s55();
        //S56
        swap(new NameSpace());
        //S57
        s57();
        //S58
        s58();
        //S59
        try {
            isReadOnly();
        } catch (HsqlException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S60
        s60();
        //S61
        s61();
        //S62
        s62();
        //S63
        getDescription();
        //S64
        s64();
        //S65
        s65();
        //S66
        s66();
        //S67
        addZero(1);
        //S68
        run(new RunNotifier());
        //S69
        try {
            s69();
        } catch (SQLException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        //S70
        mousePressed(new MouseEvent());
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
        actionPerformed2(new ActionEvent(new Tasks("message"), 1, "blink"));
        //S97
        getButtonAction();
        //S98
        s98();
        //S99
        s99();
        //S100
        purgeOldMessagesFromMessagesToIgnore(1);
    }

    // Snippet s1
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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

    // Snippet s3
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public void quit() {
        getConnectController().quitGame(true);
        if (!windowed) {
            gd.setFullScreenWindow(null);
        }
        System.exit(0);
    }   // had to be added to allow compilation

    // Snippet s6
    //SNIPPET_STARTS
    private boolean s6() {
        xsp = jj_scanpos;
        if (jj_scan_token(100)) {
            jj_scanpos = xsp;
            if (jj_scan_token(101)) return true;
        } // had to be added to allow compilation
        return true; // had to be added to allow compilation
    }

    // Snippet s7
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public void TestMethodRunner(Object test, Method method, RunNotifier notifier, Description description) {
        super1(test.getClass(), Before.class, After.class, test); // super() renamed to super1() to allow compilation
        fTest= (Ftest) test; // Type cast to Ftest to allow compilation
        fMethod= method;
    } // added to allow compilation

    // Snippet s9
    /**
     * Returns a vector containing the URI (type + path) for all the databases.
     */
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    private Object s11() throws ClassNotFoundException {
        if (clas == null)
            throw new ClassNotFoundException(
                    "Class: " + value + " not found in namespace");

        asClass = clas;
        return asClass;
    }

    // Snippet s12
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    Object removeName (String name) throws HsqlException {

        Object owner = nameList.remove(name);

        if (owner == null) {

            // should contain name
            throw Trace.error(Trace.GENERAL_ERROR);
        }

        return owner;
    }// added to allow compilation

    // Snippet s15
    //SNIPPET_STARTS
    public static void s15() {
        int stepSize = Math.min((option.getMaximumValue() - option.getMinimumValue()) / 10, 1000);
        spinner = new JSpinner(new SpinnerNumberModel(option.getValue(), option.getMinimumValue(),
                option.getMaximumValue(), Math.max(1, stepSize)));
        spinner.setToolTipText(option.toString()); // rename getShortDescription to toString to allow compilation
    }

    // Snippet s16
    //SNIPPET_STARTS
    public void s16() {
        if ( parent != null )
            setStrictJava( parent.getStrictJava() );
        this.sourceFileInfo = sourceFileInfo;
        BshClassManager bcm = BshClassManager.createClassManager( this );
    }

    // Snippet s17
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    void Constraint(HsqlName name, int[] mainCols, Table refTable, int[] refCols, // Added return type void to allow compilation
               int type, int deleteAction, int updateAction) {

        core              = new ConstraintCore();
        constName         = name;
        constType         = type;
    } // Added to allow compilation

    // Snippet s20
    //SNIPPET_STARTS
    public void s20() {
        int eventId = active? ON : OFF;
        ActionEvent blinkEvent = new ActionEvent(this,eventId,"blink");

        fireActionEvent(blinkEvent);
    }

    // Snippet s21
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public JPanel createSettingsPanel() {
        mPanel = new CapturePluginPanel(mOwner, mCloneData);
        mPanel.setBorder(Borders.createEmptyBorder(Sizes.DLUY5, Sizes.DLUX5, Sizes.DLUY5, Sizes.DLUX5));
        mPanel.setSelectedTab(mCurrentPanel);
        return new JPanel();                                                                                    /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s23
    //SNIPPET_STARTS
    protected void printFailures(Result result) {
        if (result.getFailureCount() == 0)
            return;
        if (result.getFailureCount() == 1)
            getWriter().println("There was " + result.getFailureCount() + " failure:");
        else
            getWriter().println("There were " + result.getFailureCount() + " failures:");
    } // Added to allow compilation

    // Snippet s24
    //SNIPPET_STARTS
    public static long getNormalisedTime(long t) {

        synchronized (tempCalDefault) {
            setTimeInMillis(tempCalDefault, t);
            resetToTime(tempCalDefault);

            return getTimeInMillis(tempCalDefault);
        }  // Added to allow compilation
    } // Added to allow compilation

    // Snippet s25
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    private Point comuteDisplayPointCentre(Dimension dim) {
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (screen.width - dim.width) / 2;
        int y = (screen.height - dim.height) / 2;
        return new Point(x, y);                                                     /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s28
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public void setMapTransform(MapTransform mt) {
        currentMapTransform = mt;
        MapControlsAction mca = (MapControlsAction) freeColClient.getActionManager().getFreeColAction(MapControlsAction.ID);
        if (mca.getMapControls() != null) {
            mca.getMapControls().update(mt);
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s31
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public void ComparisonCompactor(int contextLength, String expected, String actual) { // return type void added to allow compilation
        fContextLength = contextLength;
        fExpected = expected;
        fActual = actual;
    } // Added to allow compilation

    // Snippet s34
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public Object s37() {
        for(AbstractPluginProgramFormating config : mConfigs)
            if(config != null && config.isValid())
                list.add(new ProgramReceiveTarget(this, config.getName(), config.getId()));

        if(list.isEmpty())
            list.add(new ProgramReceiveTarget(this, DEFAULT_CONFIG.getName(), DEFAULT_CONFIG.getId()));

        return list.toArray(new ProgramReceiveTarget[list.size()]);
    }

    // Snippet s38                                                                      /*ORIGINALLY COMMENTED OUT, ALTERED METHOD*/
    //SNIPPET_STARTS
    Class<? extends Throwable> expectedException(Method method){
        Test annotation = method.getAnnotation(Test.class);
        if (annotation.expected() == None.class)
            return null;
        else
            return annotation.expected();
    }

    // Snippet s39
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    private Element deliverGift(Element element) {
        Element unitElement = Message.getChildElement(element, Unit.getXMLElementTagName());

        Unit unit = (Unit) getGame().getFreeColGameObject(unitElement.getAttribute("ID"));
        unit.readFromXMLElement(unitElement);
        return unitElement;                                                                                 /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s41
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public class TestClassRunnerForParameters extends Tasks{ // Added class wrapper to allow compilation
        private TestClassRunnerForParameters(Class<?> klass, Object[] parameters, int i) {
            super(klass);
            fParameters= parameters;
            fParameterSetNumber= i;
        } // Added to allow compilation
    }

    // Snippet s44
    //SNIPPET_STARTS
    void link(IndexRowIterator other) {

        other.next = next;
        other.last = this;
        next.last  = other;
    } // Added to allow compilation

    // Snippet s45
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public void readData(ObjectInputStream stream) throws IOException, ClassNotFoundException {
        int version = stream.readInt();
        mNumber = stream.readInt();
        mName = stream.readUTF();
    } // Added to allow compilation

    // Snippet s48
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    private static String getBaseName( String className )
    {
        int i = className.indexOf("$");
        if ( i == -1 )
            return className;

        return className.substring(i+1);
    } // Added to allow compilation

    // Snippet s52
    //SNIPPET_STARTS
    public void s52() {

        panel.add(UiUtilities.createHelpTextArea(mLocalizer.msg("help","No endtime defined")), cc.xy(1,1));

        mTimePanel = new TimeDateChooserPanel(date);
        panel.add(mTimePanel, cc.xy(1,3));

    } // Added to allow compilation

    // Snippet s53
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public void s54() {
        // ----------------------------------------------------------------
        // required
        // ----------------------------------------------------------------
        addColumn(t, "PROCEDURE_CAT", Types.VARCHAR);
        addColumn(t, "PROCEDURE_SCHEM", Types.VARCHAR);
        addColumn(t, "PROCEDURE_NAME", Types.VARCHAR, false);    // not null
    } // Added to allow compilation

    // Snippet s55
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public NameSpace swap( NameSpace newTop ) {
        NameSpace oldTop = (NameSpace)(stack.elementAt(0));
        stack.setElementAt( newTop, 0 );
        return oldTop;
    } // Added to allow compilation

    // Snippet s57
    //SNIPPET_STARTS
    public void s57() {
        String classname = (String) in.readObject();
        String devname = (String)in.readObject();

        DeviceIf dev = DriverFactory.getInstance().createDevice(classname, devname);
    } // Added to allow compilation

    // Snippet s58
    //SNIPPET_STARTS
    public Object s58() {
        String simpleName= runnerClass.getSimpleName();
        InitializationError error= new InitializationError(String.format(
                CONSTRUCTOR_ERROR_FORMAT, simpleName, simpleName));
        return Request.errorReport(fTestClass, error).getRunner();
    } // Added to allow compilation

    // Snippet s59
    //SNIPPET_STARTS
    public boolean isReadOnly() throws HsqlException {

        Object info = getAttribute(Session.INFO_CONNECTION_READONLY);

        isReadOnly = ((Boolean) info).booleanValue();

        return isReadOnly;
    } // Added to allow compilation

    // Snippet s60
    //SNIPPET_STARTS
    public Object s60() {
        boolean response = warehouseDialog.getResponseBoolean();
        remove(warehouseDialog);
        return response;
    } // Added to allow compilation

    // Snippet s61
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public void s62() {
        String channelId;

        if (version==1) {
            dataServiceId = (String) in.readObject();
            channelId = "" + in.readInt();
        }
    } // Added to allow compilation

    // Snippet s63
    // @Override // Removed to allow compilation
    //SNIPPET_STARTS
    public Description getDescription() {
        Description spec = Description.createSuiteDescription(getName());
        List<Method> testMethods = fTestMethods;
        for (Method method : testMethods)
            spec.addChild(methodDescription(method));

        return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s64
    //SNIPPET_STARTS
    public void s64() {
        while (classNames.hasNext()) {
            clsName = (String) classNames.next();
            clsCat = ns.getCatalogName(clsName);
            clsSchem = ns.getSchemaName(clsName);
        }
    } // Added to allow compilation

    // Snippet s65
    //SNIPPET_STARTS
    public void s65() {
        String[] texts = new String[messages.length];
        ImageIcon[] images = new ImageIcon[messages.length];
        for (int i = 0; i < messages.length; i++) {
            String ID = messages[i].getMessageID();
        }
    } // Added to allow compilation

    // Snippet s66
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
    public void run(RunNotifier notifier) {
        TestResult result= new TestResult();
        result.addListener(createAdaptingListener(notifier));
        fTest.run(result);
    } // Added to allow compilation


    // Snippet s69
    //SNIPPET_STARTS
    public void s69() throws SQLException {
        t.checkColumnsMatch(tc.core.mainColArray, tc.core.refTable,
                tc.core.refColArray);
        session.commit();

        TableWorks tableWorks = new TableWorks(session, t);
    } // Added to allow compilation

    // Snippet s70
//    @Override // Removed to allow compilation
    //SNIPPET_STARTS
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
    //SNIPPET_STARTS
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
    //private static Class<?>[] getAnnotatedClasses(Class<?> klass) throws InitializationError {
    //        SuiteClasses annotation= klass.getAnnotation(SuiteClasses.class);
    //        if (annotation == null)
    //            throw new InitializationError(String.format("class '%s' must have a SuiteClasses annotation", klass.getName()));
    //        return annotation.value();
    //} // Added to allow compilation

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
    //@SuppressWarnings("all")
    //public abstract class SuiteClasses {
    //    public SuiteClasses getAnnotation() {
    //        return null;
    //    }

    //    public Class<?>[] value() {
    //        return null;
    //    }
    //}

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
        public static BshClassManager createClassManager(Tasks tasks) {
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
        public ActionEvent(Tasks tasks, int eventId, String blink) {
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
    public Tasks(FreeColClient freeColClient, String s, Object o, KeyStroke d) {

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
        public ProgramReceiveTarget(Tasks tasks, Object name, Object id) {
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
        public Tasks last;
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
    private class InitializationError {
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

