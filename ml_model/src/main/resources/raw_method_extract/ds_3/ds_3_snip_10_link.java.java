// Snippet s44
// SNIPPET_STARTS
void link(IndexRowIterator other) {
    other.next = next;
    other.last = this;
    next.last = other;
}