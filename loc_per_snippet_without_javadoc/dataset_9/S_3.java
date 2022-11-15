    public void logAndEmailSeriousProblemS121(Throwable ex, HttpServletRequest aRequest)
    {
    /* Create trouble ticket with context reference. */
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    /* Log message to file. */
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    /* Log message to output. */
    System.out.println("SERIOUS PROBLEM OCCURRED."); // changed to allow compilation
    System.out.println(troubleTicket.toString()); // changed to allow compilation
    /* Remember most recent ticket and inform webmaster. */
    HttpSession session = aRequest.getSession();
    ServletContext context = session.getServletContext();
    context.setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    troubleTicket.toString(); // changed to allow compilation
    }
