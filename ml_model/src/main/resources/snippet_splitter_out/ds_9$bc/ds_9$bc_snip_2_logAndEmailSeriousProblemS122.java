package FeatureExtraction.snippet_splitter_out;
public class ds_9$bc_snip_2_logAndEmailSeriousProblemS122 {
// SNIPPET_END_1
// S1_2:2 resolved method chains, bad comments
/**
 * Informs the webmaster of an unexpected problem (Exception “ex”)
 * with the deployed application (indicated by “aRequest”).
 */
// SNIPPET_STARTS_2
public void logAndEmailSeriousProblemS122(Throwable ex, HttpServletRequest aRequest) {
    /* Define local variable. */
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    /* Log message. */
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    /* Log message again. */
    // changed to allow compilation
    System.out.println("SERIOUS PROBLEM OCCURRED.");
    // changed to allow compilation
    System.out.println(troubleTicket.toString());
    /* Update context and mail trouble ticket. */
    HttpSession session = aRequest.getSession();
    ServletContext context = session.getServletContext();
    context.setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    // changed to allow compilation
    troubleTicket.toString();
}
}