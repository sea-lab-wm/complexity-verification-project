package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_81_importPackage {
// Added to allow compilation
// Snippet s81
/**
 *     subsequent imports override earlier ones
 */
// SNIPPET_STARTS
public void importPackage(String name) {
    if (importedPackages == null)
        importedPackages = new Vector();
    // If it exists, remove it and add it at the end (avoid memory leak)
    if (importedPackages.contains(name))
        importedPackages.remove(name);
    importedPackages.addElement(name);
}
}