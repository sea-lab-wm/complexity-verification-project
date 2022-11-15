    private Object s11() throws ClassNotFoundException {
        if (clas == null)
            throw new ClassNotFoundException(
                    "Class: " + value + " not found in namespace");

        asClass = clas;
        return asClass;
    }
