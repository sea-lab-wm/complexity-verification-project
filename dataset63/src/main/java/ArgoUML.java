import java.awt.Dimension;
import java.awt.Rectangle;
import java.beans.PropertyChangeEvent;
import java.beans.PropertyChangeListener;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import java.io.Reader;
import java.io.StringReader;

import org.xml.sax.InputSource;
import java.util.Deque;
import java.util.HashMap;
import java.util.ArrayDeque;

import javax.swing.JTextField;
import javax.swing.JPanel;

class StubSuperClass {
    public void addNodeRelatedEdges(Object node) { 
    }
}

public class ArgoUML extends StubSuperClass {

    //SNIPPET_STARTS
    public final boolean saveDefault(boolean force) {
        if (force) {
            File toFile = new File(getDefaultPath());
            boolean saved = saveFile(toFile);
            if (saved) {
                loadedFromFile = toFile;
            }
            return saved;
        }
        if (!loaded) {
            return false;
        }

        if (loadedFromFile != null) {
            return saveFile(loadedFromFile);
        }
        if (loadedFromURL != null) {
            return saveURL(loadedFromURL);
        }
        return false;
    }
    

    //SNIPPET_STARTS
    // @Override // Removed to allow compilation
    public boolean canAddEdge(Object edge)  {
        if (edge == null) {
            return false;
        }
        if (containsEdge(edge)) {
            return false;
        }
        Object end0 = null, end1 = null;
        if (edge instanceof CommentEdge) {
            end0 = ((CommentEdge) edge).getSource();
            end1 = ((CommentEdge) edge).getDestination();
        } else if (Model.getFacade().isAAssociationEnd(edge)) {
            end0 = Model.getFacade().getAssociation(edge);
            end1 = Model.getFacade().getType(edge);

            return (end0 != null
                    && end1 != null
                    && (containsEdge(end0) || containsNode(end0))
                    && containsNode(end1));
        } else if (Model.getFacade().isARelationship(edge)) {
            end0 = Model.getCoreHelper().getSource(edge);
            end1 = Model.getCoreHelper().getDestination(edge);
        } else if (Model.getFacade().isALink(edge)) {
            end0 = Model.getCommonBehaviorHelper().getSource(edge);
            end1 =
                    Model.getCommonBehaviorHelper().getDestination(edge);
        } else if (edge instanceof CommentEdge) {
            end0 = ((CommentEdge) edge).getSource();
            end1 = ((CommentEdge) edge).getDestination();
        } else {
            return false;
        }

        if (end0 == null || end1 == null) {
            LOG.error("Edge rejected. Its ends are not attached to anything");
            return false;
        }

        if (!c1ontainsNode(end0)
                && !containsEdge(end0)) {
            LOG.error("Edge rejected. Its source end is attached to "
                    + end0
                    + " but this is not in the graph model");
            return false;
        }
        if (!containsNode(end1)
                && !containsEdge(end1)) {
            LOG.error("Edge rejected. Its destination end is attached to "
                    + end1
                    + " but this is not in the graph model");
            return false;
        }

        return true;
    }
    

    //SNIPPET_STARTS
    public Iterator childIterator(Object parent) {
        List res = new ArrayList();
        if (parent instanceof Project) {
            Project p = (Project) parent;
            res.addAll(p.getUserDefinedModelList());
            res.addAll(p.getDiagramList());
        } else if (parent instanceof ArgoDiagram) {
            ArgoDiagram d = (ArgoDiagram) parent;
            res.addAll(d.getGraphModel().getNodes());
            res.addAll(d.getGraphModel().getEdges());
        } else if (Model.getFacade().isAModelElement(parent)) {
            res.addAll(Model.getFacade().getModelElementContents(parent));
        }

        return res.iterator();
    }
    

    //SNIPPET_STARTS
    public void init() {
        Object classCls = Model.getMetaTypes().getUMLClass();
        Agency.register(crConsiderSingleton, classCls);
        Agency.register(crSingletonViolatedMSA, classCls);
        Agency.register(crSingletonViolatedOPC, classCls);
    }
    

    //SNIPPET_STARTS 
    // @Override // Removed to allow compilation
    public void addNodeRelatedEdges(Object node) {
        super.addNodeRelatedEdges(node);

        if (Model.getFacade().isAClassifier(node)) {
            Collection ends = Model.getFacade().getAssociationEnds(node);
            for (Object end : ends) {
            if (canAddEdge(Model.getFacade().getAssociation(end))) {
                        addEdge(Model.getFacade().getAssociation(end));
                    }
            }
        }
        if (Model.getFacade().isAGeneralizableElement(node)) {
            Collection generalizations = 
                Model.getFacade().getGeneralizations(node);
            for (Object generalization : generalizations) {
            if (canAddEdge(generalization)) {
                addEdge(generalization);
                return;
            }
            }
            Collection specializations = Model.getFacade().getSpecializations(node);
            for (Object specialization : specializations) {
            if (canAddEdge(specialization)) {
                addEdge(specialization);
                return;
            }
            }
        }
        if (Model.getFacade().isAModelElement(node)) {
            Collection dependencies =
            new ArrayList(Model.getFacade().getClientDependencies(node));
            dependencies.addAll(Model.getFacade().getSupplierDependencies(node));
            for (Object dependency : dependencies) {
            if (canAddEdge(dependency)) {
                addEdge(dependency);
                return;
            }
            }
        }
    }
    

    //SNIPPET_STARTS
    // @Override // Removed to allow compilation
    public boolean predicate2(Object dm, Designer dsgr) {

        if (!(Model.getFacade().isAClass(dm))) {
            return NO_PROBLEM;
        }

        if (Model.getFacade().isAAssociationClass(dm)) {
            return NO_PROBLEM;
        }

        if (Model.getFacade().getName(dm) == null
                || "".equals(Model.getFacade().getName(dm))) {
            return NO_PROBLEM;
        }

        if (!(Model.getFacade().isPrimaryObject(dm))) {
            return NO_PROBLEM;
        }

        if (Model.getFacade().isAbstract(dm)) {
            return NO_PROBLEM;
        }

        if (Model.getFacade().isSingleton(dm)) {
            return NO_PROBLEM;
        }

        if (Model.getFacade().isUtility(dm)) {
            return NO_PROBLEM;
        }

        Iterator iter = Model.getFacade().getAttributes(dm).iterator();

        while (iter.hasNext()) {
            if (!Model.getFacade().isStatic(iter.next())) {
                return NO_PROBLEM;
            }
        }

        Iterator ends = Model.getFacade().getAssociationEnds(dm).iterator();

        while (ends.hasNext()) {
            Iterator otherends =
            Model.getFacade()
                .getOtherAssociationEnds(ends.next()).iterator();

            while (otherends.hasNext()) {
            if (Model.getFacade().isNavigable(otherends.next())) {
                return NO_PROBLEM;
            }
            }
        }

        return PROBLEM_FOUND;
    }
    

    //SNIPPET_STARTS
    private void addUserDefinedProfile(String fileName, StringBuffer xmi,
            ProfileManager profileManager) throws IOException {
        File profilesDirectory = getProfilesDirectory(profileManager);
        File profileFile = new File(profilesDirectory, fileName);
        OutputStreamWriter writer = new OutputStreamWriter(
                new FileOutputStream(profileFile), 
                Argo.getEncoding());
        writer.write(xmi.toString());
        writer.close();
        LOG.info("Wrote user defined profile \"" + profileFile 
            + "\", with size " + xmi.length() + ".");
        if (isSomeProfileDirectoryConfigured(profileManager)) {
            profileManager.refreshRegisteredProfiles();
        } else {
            profileManager.addSearchPathDirectory(
                profilesDirectory.getAbsolutePath());
        }
    }
    

    //SNIPPET_STARTS
    // @Override // Removed to allow compilation
    protected void setStandardBounds(int x, int y, int w, int h) {
        if (getNameFig() == null) {
            return;
        }

        Rectangle oldBounds = getBounds();

        Dimension nameMin = getNameFig().getMinimumSize();

        getBigPort().setBounds(x, y, w, h);
        cover.setBounds(x, y, w, h);
        getNameFig().setBounds(x, y, nameMin.width + 10, nameMin.height + 4);

        
        _x = x; _y = y; _w = w; _h = h;

        firePropChange("bounds", oldBounds, getBounds());
        calcBounds(); 
        updateEdges();
    }
    

    //SNIPPET_STARTS
    // @Override // Removed to allow compilation
    public void addEdge(Object edge) {
        LOG.debug("adding class edge!!!!!!");
        if (!canAddEdge(edge)) {
            return;
        }
        getEdges().add(edge);
        
        if (Model.getFacade().isAModelElement(edge)
        && Model.getFacade().getNamespace(edge) == null) {
            Model.getCoreHelper().addOwnedElement(getHomeModel(), edge);
        }
        fireEdgeAdded(edge);
    }
    

    //SNIPPET_STARTS
    protected final void removeAllElementListeners(
            java.beans.PropertyChangeListener listener) { // Changed to allow compilation
        for (Object[] lis : listeners) {
            Object property = lis[1]; // Changed (lilisteners --> lis) to allow compilation
            if (property == null) {
                Model.getPump().removeModelEventListener(listener, lis[0]);
            } else if (property instanceof String[]) {
                Model.getPump().removeModelEventListener(listener, lis[0],
                        (String[]) property);
            } else if (property instanceof String) {
                Model.getPump().removeModelEventListener(listener, lis[0],
                        (String) property);
            } else {
                throw new RuntimeException(
                        "Internal error in removeAllElementListeners");
            }
        }
        listeners.clear();
    }
    

    //SNIPPET_STARTS
    public Collection loadModel(ProfileReference reference) 
        throws ProfileException {
        
        if (reader == null) {
            LOG.error("Profile not found");
            throw new ProfileException("Profile not found!");
        }
        
        try {
            XmiReader xmiReader = Model.getXmiReader();
            InputSource inputSource = new InputSource(reader); // fixed to allow compilation (added 'new')
            inputSource.setSystemId(reference.getPath());
            inputSource.setPublicId(
                    reference.getPublicReference().toString());
            Collection elements = xmiReader.parse(inputSource, true);
            return elements;
        } catch (UmlException e) {
            throw new ProfileException("Invalid XMI data!", e);
        }
    }
    

    //SNIPPET_STARTS
    public synchronized void addCommand(Command command) {

        ProjectManager.getManager().setSaveEnabled(true);
        
        if (undoMax == 0) {
            return;
        }
        
        if (!command.isUndoable()) {
            undoStack.clear();
            newInteraction = true;
        }
        
        final Interaction macroCommand;
        if (newInteraction || undoStack.isEmpty()) {
            redoStack.clear();
            newInteraction = false;
            if (undoStack.size() > undoMax) {
                undoStack.remove(0);
            }
            macroCommand = new Interaction(newInteractionLabel);
            undoStack.push(macroCommand);
        } else {
            macroCommand = undoStack.peek();
        }
        macroCommand.addCommand(command);
    }
    

    //SNIPPET_STARTS
    public void vetoableChange(PropertyChangeEvent pce) {
        
        if ("ownedElement".equals(pce.getPropertyName())) {
            List oldOwned = (List) pce.getOldValue();
            Object eo =  pce.getNewValue();
            Object me = Model.getFacade().getModelElement(eo);
            if (oldOwned.contains(eo)) {
            LOG.debug("model removed " + me);
            if (Model.getFacade().isAClassifier(me)) {
                        removeNode(me);
                    }
            if (Model.getFacade().isAMessage(me)) {
                        removeNode(me);
                    }
            if (Model.getFacade().isAAssociation(me)) {
                        removeEdge(me);
                    }
            } else {
            LOG.debug("model added " + me);
            }
        }
    }
    

    //SNIPPET_STARTS
    private boolean isSelectedInternal(String name) {
        Map.Entry<ModuleInterface, ModuleStatus> entry = findModule(name);

        if (entry != null) {
            ModuleStatus status = entry.getValue();

            if (status == null) {
            return false;
            }

            return status.isSelected();
        }
        return false;
    }
    

    //SNIPPET_STARTS
    public Object execute() {
        final Iterator<Command> it = commands.iterator();
        while (it.hasNext()) {
            it.next().execute();
        }
        return null;
    }
    

    //SNIPPET_STARTS
    public void run() {
        int port = 0;
        try {
            port = Integer.parseInt(hostPort.getText());
            ServerSocket serverSocket = new ServerSocket(port);
            Socket s = serverSocket.accept();
            serverSocket.close();

            System.out.println("Accepted peer connection.");

            conn = ConnectionFactory.getInstance().createServerConnection(s, 0);
            
            conn.addConnectionListener(connectionListener);

            board = new Board();
            panConnect.setEnabled(false);
            panXmit.setEnabled(true);

        } catch (Throwable err) {
            err.printStackTrace();
        }
    }
    //SNIPPETS_END
    


    // STUBS ADDED BY NADEESHAN
    public static final Logger LOG = new Logger();

    public boolean containsEdge(Object edge) {
        return false;
    }
    public boolean containsNode(Object node) {
        return false;
    }
    public boolean c1ontainsNode(Object node) {
        return false;
    }
    
    public File loadedFromFile = new File("");
    public String loadedFromURL = "";
    public boolean loaded;
    
    public boolean saveURL(String url) {
        return false;
    }
    public boolean saveFile(File toFile) {
        return false;
    }
    public String getDefaultPath() {
        return "";
    }
    
    public class Project {
        public List<?> getUserDefinedModelList() {
            return new ArrayList<>();
        }
        public List<?> getDiagramList() {
            return new ArrayList<>();
        }
    }
    public class ArgoDiagram {
        GraphModel getGraphModel() {
            return new GraphModel();
        }
    }
    public class GraphModel {
        List<?> getNodes() {
            return new ArrayList<>();
        }
        List<?> getEdges() {
            return new ArrayList<>();
        }
    }
    
    public static class MetaTypes {
        Object getUMLClass() {
            return new Object();
        }
    }
    public static class Agency {
        static void register(Object rule, Object metaType) {
        }
    }

    public Object crConsiderSingleton = new Object();
    public Object crSingletonViolatedMSA = new Object();
    public Object crSingletonViolatedOPC = new Object();
    
    public class Designer {}
    public static final boolean NO_PROBLEM = false;
    public static final boolean PROBLEM_FOUND = true;
    
    public static class Argo {
        static String getEncoding() {
            return "UTF-8";
        }
    }
    public static class ProfileManager {
        void refreshRegisteredProfiles() {}
        void addSearchPathDirectory(String path) {}
    }
    public File getProfilesDirectory(ProfileManager pm) {
        return new File(".");
    }
    
    public class Fig {
        Dimension getMinimumSize() {
            return new Dimension(1,1);
        }
        void setBounds(int x, int y, int w, int h) {}
    }
    public Fig getNameFig() {
        return new Fig();
    }
    public Fig getBigPort() {
        return new Fig();
    }
    public Fig cover = new Fig();
    public Rectangle getBounds() {
        return new Rectangle(_x, _y, _w, _h);
    }
    public void firePropChange(String prop, Object oldValue, Object newValue) {}
    public void calcBounds() {}
    public void updateEdges() {}
    public int _x, _y, _w, _h;
    
    public List<Object> edges = new ArrayList<>();
    public List<Object> getEdges() { 
        return edges; 
    }
    public void fireEdgeAdded(Object edge) {}
    public Object getHomeModel() { 
        return new Object(); 
    }
    
    public final Collection<Object[]> listeners = new ArrayList<Object[]>();
    
    public static class ProfileReference {
        String getPath() { 
            return "stubPath"; 
        }
        Object getPublicReference() { 
            return "stubPublicReference"; 
        }
    }
    public static class ProfileException extends Exception {
        ProfileException(String message) { 
            super(message); 
        }
        ProfileException(String message, Throwable cause) { 
            super(message, cause); 
        }
    }
    public Reader reader = new StringReader("<stub/>");
    public static class UmlException extends Exception {};
    public static class XmiReader {
        Collection<?> parse(org.xml.sax.InputSource source, boolean flag) throws UmlException {
            return new ArrayList<>();
        }
    }

    public class Command {
        boolean isUndoable() { return true; }
        void execute() {
        }
    }
    public class Interaction extends Command {
        Interaction(String label) {}
        void addCommand(Command command) {}
    }
    public static class ProjectManager {
        static ProjectManager getManager() { return new ProjectManager(); }
        void setSaveEnabled(boolean enabled) {}
    }
    public int undoMax = 20;
    public boolean newInteraction = true;
    public String newInteractionLabel = "New Interaction";
    public Deque<Interaction> undoStack = new ArrayDeque<>();
    public Deque<Interaction> redoStack = new ArrayDeque<>();
    
    public class ModuleInterface {}
    public class ModuleStatus {
        boolean isSelected() { 
            return false; 
        }
    }
    public Map.Entry<ModuleInterface, ModuleStatus> findModule(String name) {
        return new HashMap.Entry<ModuleInterface, ModuleStatus>() {
            @Override
            public ModuleInterface getKey() {
                return new ModuleInterface();
            }
            @Override
            public ModuleStatus getValue() {
                return new ModuleStatus();
            }
            @Override
            public ModuleStatus setValue(ModuleStatus value) {
                return value;
            }
        };
    }
    
    public final List<Command> commands = new ArrayList<>();
    
    public static class Connection {
        void addConnectionListener(Object listener) {}
    }
    public static class ConnectionFactory {
        static ConnectionFactory getInstance() { return new ConnectionFactory(); }
        Connection createServerConnection(Socket s, int i) { return new Connection(); }
    }
    public class Board {}
    public Connection conn = new Connection();
    public Object connectionListener = new Object();
    public Board board = new Board();
    public JTextField hostPort = new JTextField();
    public JPanel panConnect = new JPanel();
    public JPanel panXmit = new JPanel();
    
    public void removeEdge(Object me) {}
    public void removeNode(Object me) {}
    public boolean isSomeProfileDirectoryConfigured(ArgoUML.ProfileManager profileManager) {
        return false;
    }
    public class CommentEdge {
        Object getSource() {
            return "";
        }
        Object getDestination() {
            return "";
        }
    }
    public static class Pump {
        void removeModelEventListener(PropertyChangeListener listener, Object obj, String property) {}
        void removeModelEventListener(PropertyChangeListener listener, Object obj, String[] property) {}
        void removeModelEventListener(PropertyChangeListener listener, Object obj) {}
    }
    public static class Model {
        static MetaTypes getMetaTypes() {
            return new MetaTypes();
        }
        static Facade getFacade() {
            return new Facade();
        }
        static CoreHelper getCoreHelper() {
            return new CoreHelper();
        }
        static CommonBehaviorHelper getCommonBehaviorHelper() {
            return new CommonBehaviorHelper();
        }
        static Pump getPump() {
            return new Pump();
        }
        static XmiReader getXmiReader() { 
            return new XmiReader(); 
        }
    }
    public static class Facade {
        boolean isAAssociationEnd(Object edge) {
            return false;
        }
        boolean isAAssociation(Object me) {
            return false;
        }
        boolean isAMessage(Object me) {
            return false;
        }
        Object getModelElement(Object eo) {
            return "";
        }
        Collection<?> getSupplierDependencies(Object node) {   
            return new ArrayList<>();
        }
        Collection<?> getClientDependencies(Object node) {
           return new ArrayList<>();
        }
        Collection<?> getSpecializations(Object node) { 
            return new ArrayList<>();
        }
        Collection<?> getGeneralizations(Object node) {
            return new ArrayList<>();
        }
        boolean isAGeneralizableElement(Object node) {
            return false;
        }
        boolean isAClassifier(Object node) {
            return false;
        }
        Object getAssociation(Object edge) {
            return "";
        }
        Object getType(Object edge) {
            return "";
        }
        boolean isARelationship(Object edge) {
            return false;
        }
        boolean isALink(Object edge) {
            return false;
        }
        boolean isAModelElement(Object parent) {
            return false;
        }
        Object getNamespace(Object edge) { 
            return edge; 
        }
        Collection<?> getModelElementContents(Object parent) {
            return new ArrayList<>();
        }
        boolean isAClass(Object o) { 
            return false; 
        }
        boolean isAAssociationClass(Object o) { 
            return false; 
        }
        String getName(Object o) { 
            return ""; 
        }
        boolean isPrimaryObject(Object o) { 
            return false; 
        }
        boolean isAbstract(Object o) { 
            return false; 
        }
        boolean isSingleton(Object o) { 
            return false; 
        }
        boolean isUtility(Object o) { 
            return false; 
        }
        Collection<?> getAttributes(Object o) {
            return new ArrayList<>();
        }
        boolean isStatic(Object o) { 
            return false; 
        }
        Collection<?> getAssociationEnds(Object o) {
            return new ArrayList<>();
        }
        Collection<?> getOtherAssociationEnds(Object o) {
            return new ArrayList<>();
        }
        boolean isNavigable(Object o) { 
            return false; 
        }
    }
    public static class CoreHelper {
        Object getSource(Object edge) {
            return "";
        }
        Object getDestination(Object edge) {
            return "";
        }
        void addOwnedElement(Object homeModel, Object edge) {}
    }
    public static class CommonBehaviorHelper {
        Object getSource(Object edge) {
            return "";
        }
        Object getDestination(Object edge) {
            return "";
        }
    }
    public static class Logger {
        void debug(String msg) {}
        void info(String msg) {}
        void error(String msg) {}
    }

}
