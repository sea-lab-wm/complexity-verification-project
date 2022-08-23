    //SNIPPET_STARTS
    public void logAndEmailSeriousProblemS123(Throwable ex, HttpServletRequest aRequest)
    {
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    System.out.println("SERIOUS PROBLEM OCCURRED.");// changed to allow compilation
    System.out.println(troubleTicket.toString());// changed to allow compilation
    HttpSession session = aRequest.getSession();
    ServletContext context = session.getServletContext();
    context.setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    troubleTicket.toString();// changed to allow compilation
    }

    // Snippet 2
    // org.unicase.dashboard.impl.NotificationOperationImpl.apply
    // http://unicase.googlecode.com/svn/trunk/core/org.unicase.dashboard/src/org/unicase/dashboard/impl/Notif
    // icationOperationImpl.java

    // S2_1:1 method chains, good comments
    /**
    * Apply the transmitted notifications (“nStore”) to the project so that
    * acknowledged notifications are deleted and other ones added.
    */
