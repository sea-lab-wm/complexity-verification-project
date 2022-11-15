    public void close() {

        if (isClosed) {
            return;
        }

        isClosed = true;

        try {
            resultOut.setResultType(ResultConstants.SQLDISCONNECT);

        } finally {
            // Added to allow compilation
        }
    } // Added to allow compilation
