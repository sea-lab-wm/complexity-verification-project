    public void s74() {
        classNames = classNameSet.iterator();

        while (classNames.hasNext()) {
            className = (String) classNames.next();
            methods = iterateRoutineMethods(className, andAliases);
        } // Added to allow compilation
    } // Added to allow compilation
