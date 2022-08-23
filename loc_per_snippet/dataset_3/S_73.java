    //SNIPPET_STARTS
    public void s74() {
        classNames = classNameSet.iterator();

        while (classNames.hasNext()) {
            className = (String) classNames.next();
            methods = iterateRoutineMethods(className, andAliases);
        } // Added to allow compilation
    } // Added to allow compilation

    // Snippet s75
    /**
     * Generates a color chip image and stores it in memory.
     *
     * @param gc The GraphicsConfiguration is needed to create images that are
     *            compatible with the local environment.
     * @param c The color of the color chip to create.
     */
