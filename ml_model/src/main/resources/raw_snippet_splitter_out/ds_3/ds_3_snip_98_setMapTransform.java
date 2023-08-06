package snippet_splitter_out.ds_3;
public class ds_3_snip_98_setMapTransform {
public void setMapTransform(MapTransform mt) {
        currentMapTransform = mt;
        MapControlsAction mca = (MapControlsAction) freeColClient.getActionManager().getFreeColAction(MapControlsAction.ID);
        if (mca.getMapControls() != null) {
            mca.getMapControls().update(mt);
        } // Added to allow compilation
    }
}