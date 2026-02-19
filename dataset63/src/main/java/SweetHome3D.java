
import javax.swing.JComponent;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;

import java.awt.Component;
import java.beans.PropertyChangeSupport;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.awt.geom.Area;
import java.awt.geom.PathIterator;

import javax.swing.JSeparator;
import javax.swing.JLabel; 
import javax.swing.JSpinner;
import javax.swing.JCheckBox;


class SweetHome3DBase  extends JComponent{
    public String showSaveDialog(Object parentView,
                                 String dialogTitle,
                                 SweetHome3D.ContentType contentType,
                                 String name) {
        return name;
    }
}

public class SweetHome3D extends SweetHome3DBase {
    
    // @Override // Removed to allow compilation
    /*************   Method 20   *************/ 
    //SNIPPET_STARTS
    public String showSaveDialog(View        parentView,
                                String      dialogTitle,
                                ContentType contentType,
                                String      name) {
        if (contentType == ContentType.SWEET_HOME_3D) {
            String message = this.preferences.getLocalizedString(
                AppletContentManager.class, "showSaveDialog.message");
            String savedName = (String)JOptionPane.showInputDialog(SwingUtilities.getRootPane((JComponent)parentView), 
                message, getFileDialogTitle(true), JOptionPane.QUESTION_MESSAGE, null, null, name);
            if (savedName == null) {
            return null;
            }
            savedName = savedName.trim();

            try {
            
            if (this.recorder.exists(savedName)
                && !confirmOverwrite(parentView, savedName)) {
                return showSaveDialog(parentView, dialogTitle, contentType, savedName);
            
            } else if (savedName.length() == 0) {
                return showSaveDialog(parentView, dialogTitle, contentType, savedName);
            }
            return savedName;
            } catch (RecorderException ex) {
            String errorMessage = this.preferences.getLocalizedString(
                AppletContentManager.class, "showSaveDialog.checkHomeError");
            showError(parentView, errorMessage);
            return null;
            }
        } else {
            return super.showSaveDialog(parentView, dialogTitle, contentType, name);
        }
    }
    
    /*************   Method 21   *************/ 
    //SNIPPET_STARTS
    private void computeBounds(Node node, BoundingBox bounds, 
                                Transform3D parentTransformations, boolean transformShapeGeometry) {
        if (node instanceof Group) {
        if (node instanceof TransformGroup) {
            parentTransformations = new Transform3D(parentTransformations);
            Transform3D transform = new Transform3D();
            ((TransformGroup)node).getTransform(transform);
            parentTransformations.mul(transform);
        }
        
        Enumeration<?> enumeration = ((Group)node).getAllChildren();
        while (enumeration.hasMoreElements ()) {
            computeBounds((Node)enumeration.nextElement(), bounds, parentTransformations, transformShapeGeometry);
        }
        } else if (node instanceof Link) {
        computeBounds(((Link)node).getSharedGroup(), bounds, parentTransformations, transformShapeGeometry);
        } else if (node instanceof Shape3D) {
        Shape3D shape = (Shape3D)node;
        Bounds shapeBounds;
        if (transformShapeGeometry) {
            shapeBounds = computeTransformedGeometryBounds(shape, parentTransformations);
        } else {
            shapeBounds = shape.getBounds();
            shapeBounds.transform(parentTransformations);
        }
        bounds.combine(shapeBounds);
        }
    }
    
    /*************   Method 22   *************/ 
    //SNIPPET_STARTS
    private GraphicsConfigTemplate3D createGraphicsConfigurationTemplate3D() {
        if (System.getProperty("j3d.implicitAntialiasing") == null) {
            System.setProperty("j3d.implicitAntialiasing", "true");
        }
        GraphicsConfigTemplate3D template = new GraphicsConfigTemplate3D();

        template.setSceneAntialiasing(GraphicsConfigTemplate3D.PREFERRED);

        String stereo = System.getProperty("j3d.stereo");
        if (stereo != null) {
            if ("REQUIRED".equals(stereo))
            template.setStereo(GraphicsConfigTemplate.REQUIRED);
            else if ("PREFERRED".equals(stereo))
            template.setStereo(GraphicsConfigTemplate.PREFERRED);
        }
        return template;
    }
    
    /*************   Method 23   *************/ 
    //SNIPPET_STARTS
    public float getXCenter() {
        float xMin = this.points [0][0]; 
        float xMax = this.points [0][0]; 
        for (int i = 1; i < this.points.length; i++) {
        xMin = Math.min(xMin, this.points [i][0]);
        xMax = Math.max(xMax, this.points [i][0]);
        }
        return (xMin + xMax) / 2;
    }
    
    /*************   Method 24   *************/ 
    //SNIPPET_STARTS
    private void getAreaPoints(Area area, 
                                boolean reversed, 
                                List<float [][]> areaPoints,
                                Map<Integer, List<float [][]>> areaHoles) {

        List<float []> currentPathPoints = new ArrayList<float[]>();
        float [] previousRoomPoint = null;
        int i = 0;
        for (PathIterator it = area.getPathIterator(null, 1); !it.isDone(); it.next()) {
            float [] roomPoint = new float[2];
            switch (it.currentSegment(roomPoint)) {
            case PathIterator.SEG_MOVETO :
            case PathIterator.SEG_LINETO : 
                if (previousRoomPoint == null
                    || roomPoint [0] != previousRoomPoint [0] 
                    || roomPoint [1] != previousRoomPoint [1]) {
                currentPathPoints.add(roomPoint);
                }
                previousRoomPoint = roomPoint;
                break;
            case PathIterator.SEG_CLOSE :
                if (currentPathPoints.get(0) [0] == previousRoomPoint [0] 
                    && currentPathPoints.get(0) [1] == previousRoomPoint [1]) {
                currentPathPoints.remove(currentPathPoints.size() - 1);
                }
                if (currentPathPoints.size() > 2) {
                float [][] pathPoints = 
                    currentPathPoints.toArray(new float [currentPathPoints.size()][]);
                Room subRoom = new Room(pathPoints);
                if (subRoom.getArea() > 0) {
                    boolean pathPointsClockwise = subRoom.isClockwise();
                    if (pathPointsClockwise) {
                    
                    if (!reversed) {
                        pathPoints = getReversedArray(pathPoints);
                    }
                    List<float [][]> holes = areaHoles.get(i);
                    if (holes == null) {
                        holes = new ArrayList<float [][]>(1);
                        areaHoles.put(i, holes);
                    }
                    holes.add(pathPoints);
                    } else {
                    if (reversed) {
                        pathPoints = getReversedArray(pathPoints);
                    }
                    areaPoints.add(pathPoints);
                    i++;
                    }
                }
                }
                currentPathPoints.clear();
                previousRoomPoint = null;
                break;
            }
        }
    }
    
    /*************   Method 25   *************/
    //SNIPPET_STARTS
    private void createSwtMenu(Shell shell, UserPreferences preferences,
                                Menu menuBar, AbstractMenuItem menuItem) {
        String menuName = menuItem.getId();
        if (menuName != null) {
            AbstractMenuItem [] subMenuItems = menuItem.getChildren();
            MenuItem menuHeader;
            if (subMenuItems != null) {
            Menu currentMenu = new Menu(shell, SWT.DROP_DOWN);
            menuHeader = new MenuItem(menuBar, SWT.CASCADE);
            menuHeader.setMenu(currentMenu);
            for (AbstractMenuItem subMenuItem : subMenuItems) {
                createSwtMenu(shell, preferences, currentMenu, subMenuItem);
            }
            } else {
            menuHeader = new MenuItem(menuBar, SWT.PUSH);
            }
            menuHeader.setText(getMenuLabel(menuName, preferences));
            Image image = getIcon(shell, preferences, menuName);
            if(image != null) {
            menuHeader.setImage(image);
            }
        }
        else {
            new MenuItem(menuBar, SWT.SEPARATOR);
        }
    }
    
    /*************   Method 26   *************/
    //SNIPPET_STARTS
    private String getOptionalString(UserPreferences preferences, 
                                    Class<?> resourceClass, 
                                    String propertyKey,
                                    boolean label) {
        try {
            String localizedText = label 
                ? SwingTools.getLocalizedLabelText(preferences, resourceClass, propertyKey)
                : preferences.getLocalizedString(resourceClass, propertyKey);
            if (localizedText != null && localizedText.length() > 0) {
            return localizedText;
            } else {
            return null;
            }
        } catch (IllegalArgumentException ex) {
            return null;
        }
    }
    
    /*************   Method 60   *************/
    //SNIPPET_STARTS
    private List<HelpDocument> searchInHelpDocuments(URL helpIndex, String [] searchedWords) {
        List<URL> parsedDocuments = new ArrayList<URL>(); 
        parsedDocuments.add(helpIndex);

        List<HelpDocument> helpDocuments = new ArrayList<HelpDocument>();

        for (int i = 0; i < parsedDocuments.size(); i++) {
            try {
            
            URL helpDocumentUrl = parsedDocuments.get(i);
            HelpDocument helpDocument = new HelpDocument(helpDocumentUrl, searchedWords);
            helpDocument.parse();
            
            if (helpDocument.getRelevance() > 0) {
                helpDocuments.add(helpDocument);
            }
            
            for (URL url : helpDocument.getReferencedDocuments()) {
                String lowerCaseFile = url.getFile().toLowerCase();
                if (lowerCaseFile.endsWith(".html")
                    && !parsedDocuments.contains(url)) {
                parsedDocuments.add(url);
                } 
            } 
            } catch (IOException ex) {
            }
        }
        Collections.sort(helpDocuments, new Comparator<HelpDocument>() {
            public int compare(HelpDocument document1, HelpDocument document2) {
                return document2.getRelevance() - document1.getRelevance();
            }
            });
        return helpDocuments;
    }
    
    /*************   Method 61   *************/ 
    //SNIPPET_STARTS
    private void setLength(Float length, boolean updateEndPoint) {
        if (length != this.length) {
            Float oldLength = this.length;
            this.length = length;
            this.propertyChangeSupport.firePropertyChange(Property.LENGTH.name(), oldLength, length);
            
            if (updateEndPoint) {
            Float xStart = getXStart();
            Float yStart = getYStart();
            Float xEnd = getXEnd();
            Float yEnd = getYEnd();
            if (xStart != null && yStart != null && xEnd != null && yEnd != null && length != null) {
                if (getArcExtentInDegrees() != null && getArcExtentInDegrees().floatValue() == 0) {
                double wallAngle = Math.atan2(yStart - yEnd, xEnd - xStart);
                setXEnd((float)(xStart + length * Math.cos(wallAngle)));
                setYEnd((float)(yStart - length * Math.sin(wallAngle)));
                } else {
                throw new UnsupportedOperationException(
                    "Computing end point of a round wall from its length not supported");
                }
            } else {
                setXEnd(null);
                setYEnd(null);
            }
            }
        }
    }
    
    /*************   Method 62   *************/
    //SNIPPET_STARTS
    private void setBackFaceNormalFlip(Node node, boolean backFaceNormalFlip) {
        if (node instanceof Group) {
        
        Enumeration<?> enumeration = ((Group)node).getAllChildren(); 
        while (enumeration.hasMoreElements()) {
            setBackFaceNormalFlip((Node)enumeration.nextElement(), backFaceNormalFlip);
        }
        } else if (node instanceof Link) {
        setBackFaceNormalFlip(((Link)node).getSharedGroup(), backFaceNormalFlip);
        } else if (node instanceof Shape3D) {
        Appearance appearance = ((Shape3D)node).getAppearance();
        if (appearance == null) {
            appearance = createAppearanceWithChangeCapabilities();
            ((Shape3D)node).setAppearance(appearance);
        }
        PolygonAttributes polygonAttributes = appearance.getPolygonAttributes();
        if (polygonAttributes == null) {
            polygonAttributes = createPolygonAttributesWithChangeCapabilities();
            appearance.setPolygonAttributes(polygonAttributes);
        }
        polygonAttributes.setBackFaceNormalFlip(backFaceNormalFlip);
        }
    }
    
    /*************   Method 63   *************/
    //SNIPPET_STARTS  
    public void add(TexturesCategory category, CatalogTexture texture) {
        int index = this.categories.indexOf(category);
        
        if (index == -1) {
        category = new TexturesCategory(category.getName());
        add(category);
        } else {
        category = this.categories.get(index);
        }    
        
        category.add(texture);
        
        this.texturesChangeSupport.fireCollectionChanged(texture, 
            category.getIndexOfTexture(texture), CollectionEvent.Type.ADD);
    }
    
    /*************   Method 64   *************/
    //SNIPPET_STARTS
    private void updateAdvancedComponents() {
        Component root = SwingUtilities.getRoot(this);
        if (root != null) {
        boolean highQuality = controller.getQuality() >= 2;
        boolean advancedComponentsVisible = this.advancedComponentsSeparator.isVisible();
        if (advancedComponentsVisible != highQuality) {
            int componentsHeight = this.advancedComponentsSeparator.getPreferredSize().height + 6
                + this.dateSpinner.getPreferredSize().height + 5
                + this.ceilingLightEnabledCheckBox.getPreferredSize().height;
            this.advancedComponentsSeparator.setVisible(highQuality);
            this.dateLabel.setVisible(highQuality);
            this.dateSpinner.setVisible(highQuality);
            this.timeLabel.setVisible(highQuality);
            this.timeSpinner.setVisible(highQuality);
            this.dayNightLabel.setVisible(highQuality);
            this.ceilingLightEnabledCheckBox.setVisible(highQuality);
            root.setSize(root.getWidth(), 
                root.getHeight() + (advancedComponentsVisible ? -componentsHeight : componentsHeight));
        }
        }   
    }
    //SNIPPETS_END
    


    // STUBS ADDED BY NADEESHAN
    public HomeRecorder recorder = new HomeRecorder();
    public UserPreferences preferences = new UserPreferences();
    
    public java.beans.PropertyChangeSupport propertyChangeSupport = new PropertyChangeSupport(new Object());
    
    public class View extends JComponent {}
    public class AppletContentManager {}

    public class RecorderException extends Exception {
        public static final long serialVersionUID = 1L;
    }
    public class HomeRecorder {
        boolean exists(String name) throws RecorderException {
            return false;
        }
    }
    public class UserPreferences {
        String getLocalizedString(Class<?> cls, String key) {
            return "";
        }
    }
    public String getFileDialogTitle(boolean save) {
        return "";
    }
    public boolean confirmOverwrite(View parentView, String name) {
        return false;
    }
    public void showError(View parentView, String message) {
    }
    enum ContentType {
        SWEET_HOME_3D
    }

    public class  Node{}
    public class  Group extends Node{
        Enumeration<?> getAllChildren() {
            return Collections.emptyEnumeration();
        }
    }
    public class  TransformGroup extends Group{
        void getTransform(Transform3D transform) {
        }
    }
    public class  Link extends Node{
        Group getSharedGroup() {
            return new Group();
        }
    }
    public class  Shape3D extends Node{
        Bounds getBounds() {
            return new Bounds();
        }

        Appearance getAppearance() {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'getAppearance'");
        }

        public void setAppearance(SweetHome3D.Appearance appearance) {}
    }
    public class  Bounds{
        public void transform(Transform3D transform) {
        }
    }
    public class  BoundingBox extends Bounds{
        public void combine(Bounds bounds) {
        }
    }
    public class  Transform3D{
        public Transform3D() {
        }
        public Transform3D(Transform3D t3d) {
        }
        public void mul(Transform3D t3d) {
        }
    }

    public Bounds computeTransformedGeometryBounds(Shape3D shape, Transform3D transform) {
        return new Bounds();
    }

    public class Appearance {
        PolygonAttributes getPolygonAttributes() {
            // TODO Auto-generated method stub
            throw new UnsupportedOperationException("Unimplemented method 'getPolygonAttributes'");
        }
        public void setPolygonAttributes(SweetHome3D.PolygonAttributes polygonAttributes) {}
    }
    public class PolygonAttributes {
        void setBackFaceNormalFlip(boolean backFaceNormalFlip) {}
    }

    public Appearance createAppearanceWithChangeCapabilities() {
        return new Appearance();
    }
    public PolygonAttributes createPolygonAttributesWithChangeCapabilities() {
        return new PolygonAttributes();
    }

    public class CatalogTexture {
    }
    public List<TexturesCategory> categories = new ArrayList<>();

    public void add(TexturesCategory category) {
        this.categories.add(category);
    }

    public class TexturesCategory {
        public String name;

        TexturesCategory(String name) {
            this.name = name;
        }

        String getName() {
            return name;
        }

        void add(CatalogTexture texture) {
        }

        int getIndexOfTexture(CatalogTexture texture) {
            return 0;
        }
    }

    public static class CollectionEvent {
        enum Type {
            ADD
        }
    }

    public class CollectionChangeSupport<T> {
        void fireCollectionChanged(T element, int index, CollectionEvent.Type type) {
        }
    }

    public CollectionChangeSupport<CatalogTexture> texturesChangeSupport =
        new CollectionChangeSupport<>();

    

    public class Controller {
        public int getQuality() {
            return 0;
        }
    }

    // public class SwingUtilities {
    //     public static Component getRoot(Component comp) {
    //         return comp;
    //     }
    // }
     
    public Controller controller = new Controller();
    public JSeparator advancedComponentsSeparator = new JSeparator();
    public JLabel dateLabel = new JLabel();
    public JSpinner dateSpinner = new JSpinner();
    public JLabel timeLabel = new JLabel();
    public JSpinner timeSpinner = new JSpinner();
    public JLabel dayNightLabel = new JLabel();
    public JCheckBox ceilingLightEnabledCheckBox = new JCheckBox();
    public float [][] points = new float[][] {{ 0f, 0f }};
    
    public Float length = 0f;
   

    public enum Property {
        LENGTH
    }

    public Float getXStart() {
        throw new UnsupportedOperationException("Unimplemented method");
    }

    public Float getYStart() {
        throw new UnsupportedOperationException("Unimplemented method");
    }

    public Float getXEnd() {
        throw new UnsupportedOperationException("Unimplemented method");
    }

    public Float getYEnd() {
        throw new UnsupportedOperationException("Unimplemented method");
    }
    
    public Float getArcExtentInDegrees() {
        throw new UnsupportedOperationException("Unimplemented method");
    }

    public void setXEnd(Float xEnd) {
        throw new UnsupportedOperationException("Unimplemented method");
    }

    public void setYEnd(Float yEnd) {
        throw new UnsupportedOperationException("Unimplemented method");
    }

    public class GraphicsConfigTemplate {
        public static final int REQUIRED = 1;
        public static final int PREFERRED = 2;
    }

    public class GraphicsConfigTemplate3D extends GraphicsConfigTemplate {
        public void setSceneAntialiasing(int value) {
        }
        public void setStereo(int value) {
        }
    }

    public class Room {
        Room(float [][] points) {
        }
        double getArea() {
            return 0;
        }
        boolean isClockwise() {
            return true;
        }
    }

    public float [][] getReversedArray(float [][] array) {
        float [][] reversedArray = new float [array.length][];
        return reversedArray;
    }

    public class Shell {
    }

    public class Menu {
        static final int DROP_DOWN = 1;
        Menu(Shell shell, int style) {}
    }

    public class MenuItem {
        public static final int CASCADE = 1;
        public static final int PUSH = 2;
        public static final int SEPARATOR = 4;
        public MenuItem(Menu parent, int style) {}
        public void setMenu(Menu menu) {}
        public void setText(String text) {}
        public void setImage(Image image) {}
    }

    public abstract class AbstractMenuItem {
        abstract String getId();
        abstract AbstractMenuItem [] getChildren();
    }

    public class Image {}

    public String getMenuLabel(String menuName, UserPreferences preferences) {
        return menuName;
    }

    public Image getIcon(Shell shell, UserPreferences preferences, String menuName) {
        return new Image();
    }

    public class SWT {
        static final int DROP_DOWN = 1;
        static final int CASCADE = 2;
        static final int PUSH = 3;
        static final int SEPARATOR = 4;
    }

    public static class SwingTools {
        static String getLocalizedLabelText(UserPreferences preferences, 
                                                   Class<?> resourceClass, 
                                                   String propertyKey) {
            return preferences.getLocalizedString(resourceClass, propertyKey);
        }
    }

    public class HelpDocument {
        HelpDocument(URL url, String[] searchedWords) {}
        void parse() throws IOException {}
        int getRelevance() {
            return 0;
        }
        List<URL> getReferencedDocuments() {
            return new ArrayList<URL>();
        }
    }

    public class URL {
        String getFile() {
            return "";
        }
    }
}