package FeatureExtraction.snippet_splitter_out;
public class ds_3_snip_44_link {
// Snippet s44
// SNIPPET_STARTS
void link(IndexRowIterator other) {
    other.next = next;
    other.last = this;
    next.last = other;
}
}