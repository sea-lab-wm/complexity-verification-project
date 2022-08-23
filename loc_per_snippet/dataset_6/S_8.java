    //SNIPPET_STARTS
    public static <T> Iterator<T> limit(final Iterator<? extends T> base, final CountingPredicate<? super T> filter) {
        return new Iterator<T>() {
            private T next;
            private boolean end;
            private int index=0;
            public boolean hasNext() {
                fetch();
                return next!=null;
            }

            public T next() {
                fetch();
                T r = next;
                next = null;
                return r;
            }

            private void fetch() {
                if (next==null && !end) {
                    if (base.hasNext()) {
                        next = base.next();
                        if (!filter.apply(index++,next)) {
                            next = null;
                            end = true;
                        }
                    } else {
                        end = true;
                    }
                }
            }

            public void remove() {
                throw new UnsupportedOperationException();
            }
        };
    }

    // jenkins.util.AntClassLoader.loadClass(java.lang.String,boolean)
    /**
     * Loads a class with this class loader.
     *
     * This class attempts to load the class in an order determined by whether
     * or not the class matches the system/loader package lists, with the
     * loader package list taking priority. If the classloader is in isolated
     * mode, failure to load the class in this loader will result in a
     * ClassNotFoundException.
     *
     * @param classname The name of the class to be loaded.
     *                  Must not be <code>null</code>.
     * @param resolve <code>true</code> if all classes upon which this class
     *                depends are to be loaded.
     *
     * @return the required Class object
     *
     * @exception ClassNotFoundException if the requested class does not exist
     * on the system classpath (when not in isolated mode) or this loader's
     * classpath.
     */
