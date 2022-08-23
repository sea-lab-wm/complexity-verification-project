    //SNIPPET_STARTS
    public void Date() { // Return type void added to allow compilation
        Calendar mCalendar = Calendar.getInstance();
        mYear = mCalendar.get(Calendar.YEAR);
        mMonth = mCalendar.get(Calendar.MONTH) + 1;
    } // Added to allow compilation
