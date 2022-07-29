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
public class Tasks_1 {

    @SuppressWarnings("all")
    private static final String RETURN = "";
    @SuppressWarnings("all")
    private static final int IRETURN = 0;
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
    private int constType;
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
    private String granteeName;
    @SuppressWarnings("all")
    private GranteeManager granteeManager;
    @SuppressWarnings("all")
    private Method fMethod;
    @SuppressWarnings("all")
    private Object clas;
    @SuppressWarnings("all")
    private String value;
    @SuppressWarnings("all")
    private Object asClass;
    @SuppressWarnings("all")
    private Object cancel;
    @SuppressWarnings("all")
    private Object ok;
    @SuppressWarnings("all")
    private List<Object> nameList;
    @SuppressWarnings("all")
    private Object sourceFileInfo;
    @SuppressWarnings("all")
    private int num;
    @SuppressWarnings("all")
    private Vector<DeviceIf> mDevices;
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
    private Object mCurrentPanel;
    @SuppressWarnings("all")
    private Object player;
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
    private IntValueHashMap rightsMap;

    //ADDED BY KOBI
    @SuppressWarnings("all")
    private Logger logger;

    @SuppressWarnings("all")
    public Tasks_1(Class<?> klass) {

    }

    @SuppressWarnings("all")
    public Tasks_1(String message) {

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



    @SuppressWarnings("all")
    private static class GranteeManager {
    }

    @SuppressWarnings("all")
    private class Grantee {
    }

    @SuppressWarnings("all")
    public static class HsqlException extends Exception {
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
    private class Test {
        //ADDED BY KOBI
        public Class expected() {
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
    private class InGameInputHandler {
    }

    @SuppressWarnings("all")
    private class AbstractTranslet {
        @SuppressWarnings("all")
        public void setMessageHandler(InGameInputHandler inGameInputHandler) {

        }
    }

    @SuppressWarnings("all")
    private class Game {
        @SuppressWarnings("all")
        public Map getMap() {
            return null;
        }
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
    private class Runner {
        @SuppressWarnings("all")
        public Object getDescription() {
            return null;
        }
    }

    @SuppressWarnings("all")
    private class DeviceIf {
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
    private class IntValueHashMap {
    }

    @SuppressWarnings("all")
    private ConnectController getConnectController() {
        return null;
    }

    @SuppressWarnings("all")
    private class ConnectController {
        @SuppressWarnings("all")
        public void quitGame(boolean b) {

        }
    }

    @SuppressWarnings("all")
    private boolean jj_scan_token(int i) {
        return false;
    }

    @SuppressWarnings("all")
    private class HsqlName {
    }

    @SuppressWarnings("all")
    private class Table {
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
    private class Before {
    }

    @SuppressWarnings("all")
    private class After {
    }

    @SuppressWarnings("all")
    private class Ftest {
        @SuppressWarnings("all")
        public void run(TestResult result) {

        }
    }

    @SuppressWarnings("all")
    private static class Database {
        public Object collation;
    }

    @SuppressWarnings("all")
    private class NoTestsRemainException extends Exception {
    }

    
}

