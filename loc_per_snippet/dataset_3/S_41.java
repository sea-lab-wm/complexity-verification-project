    //SNIPPET_STARTS
    public synchronized void removeProgram(Program program) {
        PluginTreeNode node = findProgramTreeNode(program, false);
        if (node != null) {
            mChildNodes.remove(node);
            if (mMarker != null) {
                program.unmark(mMarker);
            } // Added to allow compilation
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s43
