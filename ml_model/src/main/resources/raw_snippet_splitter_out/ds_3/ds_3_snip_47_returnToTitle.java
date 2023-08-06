package snippet_splitter_out.ds_3;
public class ds_3_snip_47_returnToTitle {
public void returnToTitle() {
        // TODO: check if the GUI object knows that we're not
        // inGame. (Retrieve value of GUI::inGame.)  If GUI thinks
        // we're still in the game then log an error because at this
        // point the GUI should have been informed.
        closeMenus();
        removeInGameComponents();
        showMainPanel();
    }
}