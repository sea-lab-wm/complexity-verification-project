package FeatureExtraction.snippet_splitter_out;
public class ds_9$nc_snip_2_logAndEmailSeriousProblemS123 {
// SNIPPET_END_2
// S1_2:3 resolved method chains, no comments
/**
 * Informs the webmaster of an unexpected problem (Exception “ex”)
 * with the deployed application (indicated by “aRequest”).
 */
// SNIPPET_STARTS_3
public void logAndEmailSeriousProblemS123(Throwable ex, HttpServletRequest aRequest) {
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    // changed to allow compilation
    System.out.println("SERIOUS PROBLEM OCCURRED.");
    // changed to allow compilation
    System.out.println(troubleTicket.toString());
    HttpSession session = aRequest.getSession();
    ServletContext context = session.getServletContext();
    context.setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    // changed to allow compilation
    troubleTicket.toString();
}
}