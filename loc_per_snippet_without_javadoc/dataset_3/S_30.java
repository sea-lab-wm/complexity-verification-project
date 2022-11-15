    public String s31() {
        if (iterateOverMe instanceof String)
            return createEnumeration(((String) iterateOverMe).toCharArray());

        if (iterateOverMe instanceof StringBuffer)
            return createEnumeration(
                    iterateOverMe.toString().toCharArray());

        throw new IllegalArgumentException(
                "Cannot enumerate object of type " + iterateOverMe.getClass());
    }

    private String createEnumeration(char[] toCharArray) {
        return new String();                                                            /*Altered return*/
        //return null;
    }
