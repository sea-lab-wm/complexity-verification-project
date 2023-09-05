// SNIPPET_END_3
// S1_2:1 resolved method chains, good comments
/**
 * Informs the webmaster of an unexpected problem (Exception “ex”) with the deployed application
 * (indicated by “aRequest”).
 */
// SNIPPET_STARTS_1
public void logAndEmailSeriousProblemS121(Throwable ex, HttpServletRequest aRequest) {
    /* Create trouble ticket with context reference. */
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    /* Log message to file. */
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    /* Log message to output. */
    // changed to allow compilation
    System.out.println("SERIOUS PROBLEM OCCURRED.");
    // changed to allow compilation
    System.out.println(troubleTicket.toString());
    /* Remember most recent ticket and inform webmaster. */
    HttpSession session = aRequest.getSession();
    ServletContext context = session.getServletContext();
    context.setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    // changed to allow compilation
    troubleTicket.toString();
}