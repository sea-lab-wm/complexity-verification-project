package snippet_splitter_out.ds_3;
public class ds_3_snip_73_quit {
public void quit() {
        getConnectController().quitGame(true);
        if (!windowed) {
            gd.setFullScreenWindow(null);
        }
        System.exit(0);
    }
}