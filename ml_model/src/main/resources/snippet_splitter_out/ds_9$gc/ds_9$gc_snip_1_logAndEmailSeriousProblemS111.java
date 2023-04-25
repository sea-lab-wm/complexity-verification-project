package FeatureExtraction.snippet_splitter_out;
public class ds_9$gc_snip_1_logAndEmailSeriousProblemS111 {
// S1_1:1 method chains, good comments
/**
 * Informs the webmaster of an unexpected problem (Exception "ex")
 * with the deployed application (indicated by “aRequest”).
 */
// SNIPPET_STARTS_1
public void logAndEmailSeriousProblemS111(Throwable ex, HttpServletRequest aRequest) {
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
    aRequest.getSession().getServletContext().setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    // changed to allow compilation
    troubleTicket.toString();
}
}