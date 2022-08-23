    //SNIPPET_STARTS
    public void logAndEmailSeriousProblemS113(Throwable ex, HttpServletRequest aRequest)
    {
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    System.out.println("SERIOUS PROBLEM OCCURRED.");// changed to allow compilation
    System.out.println(troubleTicket.toString());// changed to allow compilation
    aRequest.getSession().getServletContext().
    setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    troubleTicket.toString(); // changed to allow compilation
    }

    // S1_2:1 resolved method chains, good comments
    /**
    * Informs the webmaster of an unexpected problem (Exception “ex”)
    * with the deployed application (indicated by “aRequest”).
    */
