    //SNIPPET_STARTS
    private Point comuteDisplayPointCentre(Dimension dim) {
        Dimension screen = Toolkit.getDefaultToolkit().getScreenSize();
        int x = (screen.width - dim.width) / 2;
        int y = (screen.height - dim.height) / 2;
        return new Point(x, y);                                                     /*Altered return*/
        //return null; // Added to allow compilation
    } // Added to allow compilation

    // Snippet s28
