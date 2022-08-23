    //SNIPPET_STARTS
    void link(IndexRowIterator other) {

        other.next = next;
        other.last = this;
        next.last  = other;
    } // Added to allow compilation

    // Snippet s45
