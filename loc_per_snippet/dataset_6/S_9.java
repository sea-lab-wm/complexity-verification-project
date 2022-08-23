    //SNIPPET_STARTS
    protected synchronized Class loadClass(String classname, boolean resolve)
            throws ClassNotFoundException {
        // 'sync' is needed - otherwise 2 threads can load the same class
        // twice, resulting in LinkageError: duplicated class definition.
        // findLoadedClass avoids that, but without sync it won't work.

        Class theClass = findLoadedClass(classname);
        if (theClass != null) {
            return theClass;
        }
        if (isParentFirst(classname)) {
            try {
                theClass = findBaseClass(classname);
                log("Class " + classname + " loaded from parent loader " + "(parentFirst)",
                        Project.MSG_DEBUG);
            } catch (ClassNotFoundException cnfe) {
                theClass = findClass(classname);
                log("Class " + classname + " loaded from ant loader " + "(parentFirst)",
                        Project.MSG_DEBUG);
            }
        } else {
            try {
                theClass = findClass(classname);
                log("Class " + classname + " loaded from ant loader", Project.MSG_DEBUG);
            } catch (ClassNotFoundException cnfe) {
                if (ignoreBase) {
                    throw cnfe;
                }
                theClass = findBaseClass(classname);
                log("Class " + classname + " loaded from parent loader", Project.MSG_DEBUG);
            }
        }
        if (resolve) {
            resolveClass(theClass);
        }
        return theClass;
    }
