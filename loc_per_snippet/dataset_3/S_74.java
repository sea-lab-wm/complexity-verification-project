    //SNIPPET_STARTS
    private void loadColorChip(GraphicsConfiguration gc, Color c) {
        BufferedImage tempImage = gc.createCompatibleImage(11, 17);
        Graphics g = tempImage.getGraphics();
        if (c.equals(Color.BLACK)) {
            g.setColor(Color.WHITE);
        }
    } // Added to allow compilation

    // Snippet s76
