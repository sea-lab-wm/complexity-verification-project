import javax.swing.AbstractAction;
import javax.swing.JButton;
import javax.swing.JPanel;
import javax.swing.JPopupMenu;
import javax.swing.JTextArea;

import org.w3c.dom.Element;
import org.w3c.dom.Document;

import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JMenuItem; import java.awt.GridBagConstraints;

import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.Insets;
import java.awt.LayoutManager;
import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import javax.swing.JRootPane;
import javax.swing.JSeparator;

public class ATunes<T> {
    
    //SNIPPET_STARTS
    private void arrangeDialog(final JTextArea textArea,
            final JPanel patternPreviewPanel,
            final JPanel availablePatternsPanel, final JButton okButton,
            final JPanel auxPanel) {
        JPanel panel = new JPanel(new GridBagLayout());
        GridBagConstraints c = new GridBagConstraints();
        c.gridx = 0;
        c.gridy = 0;
        c.weightx = 1;
        c.gridwidth = 2;
        c.fill = GridBagConstraints.HORIZONTAL;
        c.insets = new Insets(10, 10, 10, 10);
        panel.add(textArea, c);
        c.gridy = 1;
        c.insets = new Insets(5, 30, 5, 30);
        panel.add(this.firstElementLabel, c);
        c.gridy = 2;
        panel.add(this.patternComboBox, c);
        c.gridx = 0;
        c.gridy = 3;
        c.weightx = 0.7;
        c.weighty = 1;
        c.gridwidth = 1;
        c.fill = GridBagConstraints.BOTH;
        c.insets = new Insets(5, 5, 5, 5);
        panel.add(patternPreviewPanel, c);
        c.gridx = 1;
        c.weightx = 0.3;
        panel.add(availablePatternsPanel, c);
        c.gridx = 0;
        c.gridy = 4;
        c.gridwidth = 2;
        c.weighty = 0;
        c.fill = GridBagConstraints.NONE;
        c.anchor = GridBagConstraints.CENTER;
        c.insets = new Insets(10, 10, 10, 10);
        panel.add(auxPanel, c);
        add(panel);
        getRootPane().setDefaultButton(okButton);
    }
    

    //SNIPPET_STARTS
    // @Override // Removed to allow compilation
    public boolean equals(final Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        FontSettings other = (FontSettings) obj;
        if (this.font == null) {
            if (other.font != null) {
                return false;
            }
        } else if (!this.font.equals(other.font)) {
            return false;
        }
        if (this.useFontSmoothing != other.useFontSmoothing) {
            return false;
        }
        if (this.useFontSmoothingSettingsFromOs != other.useFontSmoothingSettingsFromOs) {
            return false;
        }
        return true;
    }
    

    //SNIPPET_STARTS
    ApplicationVersion getApplicationVersionFromXml(final Document xml) {
        Element element = (Element) xml.getElementsByTagName("latest").item(0);
        String date = XMLUtils.getChildElementContent(element, "date");
        int major = Integer.parseInt(XMLUtils.getChildElementContent(element,
                "majorNumber"));
        int minor = Integer.parseInt(XMLUtils.getChildElementContent(element,
                "minorNumber"));
        int revision = Integer.parseInt(XMLUtils.getChildElementContent(
                element, "revisionNumber"));
        String url = element.getAttribute("url");

        String directDownloadURL = applyVersion(
                XMLUtils.getChildElementContent(element, getElementNameForOS()),
                major, minor, revision);

        String changes = XMLUtils.getChildElementContent(element, "changes");

        return new ApplicationVersion(date, major, minor, revision,
                VersionType.FINAL, "", url, directDownloadURL, changes);
    }
    

    //SNIPPET_STARTS
    private List<Entry<T, Integer>> getElementsSorted() {
        List<Entry<T, Integer>> list = new ArrayList<Map.Entry<T, Integer>>(
                this.count.entrySet());
        Collections.sort(list, new Comparator<Entry<T, Integer>>() {
            @Override
            public int compare(final Entry<T, Integer> o1,
                    final Entry<T, Integer> o2) {
                return -o1.getValue().compareTo(o2.getValue());
            }
        });
        return list;
    }
    

    //SNIPPET_STARTS
    // @Override // Removed to allow compilation
    public JPopupMenu getTreePopupMenu() {
        if (this.deviceTreeMenu == null) {
            this.deviceTreeMenu = new JPopupMenu();
            AbstractActionOverSelectedObjects<IAudioObject> addToPlayListAction = getBeanFactory()
                    .getBean("addToPlayListFromDeviceNavigationView",
                            AddToPlayListAction.class);
            addToPlayListAction.setAudioObjectsSource(this);
            this.deviceTreeMenu.add(addToPlayListAction);

            SetAsPlayListAction setAsPlayListAction = getBeanFactory().getBean(
                    "setAsPlaylistFromDeviceNavigationView",
                    SetAsPlayListAction.class);
            setAsPlayListAction.setAudioObjectsSource(this);
            this.deviceTreeMenu.add(setAsPlayListAction);

            this.deviceTreeMenu.add(new JSeparator());

            AbstractActionOverSelectedTreeObjects<IFolder> openFolder = getBeanFactory()
                    .getBean("openFolderFromDeviceNavigationTree",
                            OpenFolderFromNavigatorTreeAction.class);
            openFolder.setTreeObjectsSource(this);
            this.deviceTreeMenu.add(openFolder);

            this.deviceTreeMenu.add(new JSeparator());
            this.deviceTreeMenu.add(new EditTagMenu(false, this,
                    getBeanFactory()));

            AbstractActionOverSelectedTreeObjects<IAlbum> editTitles = getBeanFactory()
                    .getBean("editTitlesFromDeviceViewAction",
                            EditTitlesAction.class);
            editTitles.setTreeObjectsSource(this);
            this.deviceTreeMenu.add(editTitles);
            this.deviceTreeMenu.add(new JSeparator());
            this.deviceTreeMenu.add(getBeanFactory().getBean(
                    RemoveFromDiskAction.class));
            this.deviceTreeMenu.add(new JSeparator());

            CopyToRepositoryAction copyToRepositoryAction = getBeanFactory()
                    .getBean(CopyToRepositoryAction.class);
            copyToRepositoryAction.setAudioObjectsSource(this);
            this.deviceTreeMenu.add(copyToRepositoryAction);

            this.deviceTreeMenu.add(getBeanFactory().getBean(
                    FillDeviceWithRandomSongsAction.class));
            this.deviceTreeMenu.add(new JSeparator());
            this.deviceTreeMenu.add(getBeanFactory().getBean(
                    SearchArtistAction.class));
            this.deviceTreeMenu.add(getBeanFactory().getBean(
                    SearchArtistAtAction.class));
        }
        return this.deviceTreeMenu;
    }
    

    //SNIPPET_STARTS
    private void moveToBottom(final IPlayList playList, final int[] rows) {
        int j = 0;
        for (int i = rows.length - 1; i >= 0; i--) {
            IAudioObject aux = playList.get(rows[i]);
            playList.remove(rows[i]);
            playList.add(playList.size() - j++, aux);
        }
        if (rows[rows.length - 1] < playList.getCurrentAudioObjectIndex()) {
            playList.setCurrentAudioObjectIndex(playList
                    .getCurrentAudioObjectIndex() - rows.length);
        } else if (rows[0] <= playList.getCurrentAudioObjectIndex()
                && playList.getCurrentAudioObjectIndex() <= rows[rows.length - 1]) {
            playList.setCurrentAudioObjectIndex(playList
                    .getCurrentAudioObjectIndex()
                    + playList.size()
                    - rows[rows.length - 1] - 1);
        }
    }
    

    //SNIPPET_STARTS
    private void fillCdInfo() {
        CDInfo info = this.cdda2wav.getCDInfo();
        info.setTracks(this.tracks);
        info.setDurations(this.durations);
        info.setDuration(this.totalDuration);
        info.setID(this.id);
        if (this.album != null && !this.album.equals("")) {
            info.setAlbum(this.album);
        }

        if (this.artist != null && !this.artist.equals("")) {
            info.setArtist(this.artist);
        }

        info.setTitles(this.titles);
        info.setArtists(this.artists);
        info.setComposers(this.composers);
    }
    

    //SNIPPET_STARTS
    // @Override // Removed to allow compilation
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + this.order;
        result = prime * result
                + ((this.sort == null) ? 0 : this.sort.hashCode());
        result = prime * result + (this.visible ? 1231 : 1237);
        result = prime * result + this.width;
        return result;
    }
    //SNIPPETS_END
    


    // STUBS ADDED BY NADEESHAN
    public class GridBagLayout implements LayoutManager {
        @Override
        public void addLayoutComponent(String name, Component comp) {}
        @Override
        public void layoutContainer(Container parent) {}
        @Override
        public Dimension minimumLayoutSize(Container parent) {
            return new Dimension(0, 0);
        }
        @Override
        public Dimension preferredLayoutSize(Container parent) {
            return new Dimension(0, 0);
        }
        @Override
        public void removeLayoutComponent(Component comp) {}
    }

    public JLabel firstElementLabel = new JLabel("First Element:");
    public JComboBox<String> patternComboBox = new JComboBox<>();

    public JRootPane getRootPane() {
        return new JRootPane();
    }
    public void add(JPanel panel) {}

    public class FontSettings {
        public Font font = new Font("Arial", Font.PLAIN, 12);
        public boolean useFontSmoothing;
        public boolean useFontSmoothingSettingsFromOs;
    }
    public Font font = new Font("Arial", Font.PLAIN, 12);
    public boolean useFontSmoothing = true;
    public boolean useFontSmoothingSettingsFromOs = false;

    
    public class ApplicationVersion {
        ApplicationVersion(String date, int major, int minor, int revision, ATunes.VersionType final1,
                String string, String url, String directDownloadURL, String changes) {
        }
    }
    public static class XMLUtils {
        static String getChildElementContent(Element element, String tagName) {
            return "";
        }
    }
    public enum VersionType {
        FINAL
    }
    
    public String applyVersion(String childElementContent, int major, int minor, int revision) {
        return "";
    }

    public String getElementNameForOS() {
        return "";
    }

    public Map<T, Integer> count = new HashMap<>();
    
    public JPopupMenu deviceTreeMenu = new JPopupMenu();
    public static class BeanFactory {
        <T> T getBean(String name, Class<T> cls) {
            return create(cls);
        }

        <T> T getBean(Class<T> cls) {
            return create(cls);
        }

        public static <T> T create(Class<T> cls) {
            try {
                return cls.getDeclaredConstructor().newInstance();
            } catch (Exception e) {
                throw new IllegalStateException(
                    "Cannot instantiate bean: " + cls.getName(), e);
            }
        }
    }
    public BeanFactory getBeanFactory() {
        return new BeanFactory();
    }
    public interface IAudioObject {}
    public interface IFolder {}
    public interface IAlbum {}

    public static abstract class AbstractActionOverSelectedObjects<T>
            extends AbstractAction {
        void setAudioObjectsSource(Object source) {}
    }

    public static abstract class AbstractActionOverSelectedTreeObjects<T>
            extends AbstractAction {
        void setTreeObjectsSource(Object source) {}
    }

    public static class AddToPlayListAction
        extends AbstractActionOverSelectedObjects<IAudioObject> {
            @Override
            public void actionPerformed(ActionEvent e) {}
    }

    public static class SetAsPlayListAction
        extends AbstractActionOverSelectedObjects<IAudioObject> {
            @Override
            public void actionPerformed(ActionEvent e) {}
    }

    public static class OpenFolderFromNavigatorTreeAction
        extends AbstractActionOverSelectedTreeObjects<IFolder> {
        @Override
        public void actionPerformed(ActionEvent e) {}
    }

    public static class EditTitlesAction
        extends AbstractActionOverSelectedTreeObjects<IAlbum> {
        @Override
        public void actionPerformed(ActionEvent e) {}
    }

    public static class RemoveFromDiskAction extends AbstractAction {
        @Override
        public void actionPerformed(ActionEvent e) {}
    }

    public static class CopyToRepositoryAction
        extends AbstractActionOverSelectedObjects<IAudioObject> {
        @Override
        public void actionPerformed(ActionEvent e) {}  
    }

    public static class FillDeviceWithRandomSongsAction extends AbstractAction {
        @Override
        public void actionPerformed(ActionEvent e) {}
    }

    public static class SearchArtistAction extends AbstractAction {
        @Override
        public void actionPerformed(ActionEvent e) {}
    }

    public static class SearchArtistAtAction extends AbstractAction {
        @Override
        public void actionPerformed(ActionEvent e) {}
    }

    public static class EditTagMenu extends JMenuItem {
        EditTagMenu(boolean b, Object o, BeanFactory f) {super();}
    }
    

    public interface IPlayList {
        IAudioObject get(int index);
        void remove(int index);
        void add(int index, IAudioObject audioObject);
        int size();
        int getCurrentAudioObjectIndex();
        void setCurrentAudioObjectIndex(int index);  
    }

    public class CDInfo {
        void setTracks(int tracks) {}
        void setDurations(List<Integer> durations) {}
        void setDuration(int totalDuration) {}
        void setID(String id) {}
        void setAlbum(String album) {}
        void setArtist(String artist) {}
        void setTitles(List<String> titles) {}
        void setArtists(List<String> artists) {}
        void setComposers(List<String> composers) {}
    }

    public class CDDA2WAV {
        CDInfo getCDInfo() {
            return new CDInfo();
        }
    }

    public CDDA2WAV cdda2wav = new CDDA2WAV();
    public int tracks = 0;
    public List<Integer> durations = new ArrayList<>();
    public int totalDuration = 0;
    public String id = "";
    public String album = "";
    public String artist = "";
    public List<String> titles = new ArrayList<>();
    public List<String> artists = new ArrayList<>();
    public List<String> composers = new ArrayList<>();
    
    public int order = 0;
    public String sort = "";
    public boolean visible = false;
    public int width = 0;

}