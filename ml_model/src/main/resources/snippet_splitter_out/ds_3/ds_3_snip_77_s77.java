package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_77_s77 {
// Added to allow compilation
// Snippet s77
// SNIPPET_STARTS
public void s77() throws FileNotFoundException, IOException {
    File data = new File(Plugin.getPluginManager().getTvBrowserSettings().getTvBrowserUserHome() + File.separator + "CaptureDevices" + File.separator + mCount + ".dat");
    ObjectOutputStream stream = new ObjectOutputStream(new FileOutputStream(data));
    dev.writeData(stream);
}
}