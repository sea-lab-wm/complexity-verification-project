package snippet_splitter_out.ds_9$nc;

public class ds_9$nc_snip_1_logAndEmailSeriousProblemS113 {
  // SNIPPET_END_2
  // S1_1:3 method chains, no comments
  /**
   * Informs the webmaster of an unexpected problem (Exception “ex”) with the deployed application
   * (indicated by “aRequest”).
   */
  // SNIPPET_STARTS_3
  public void logAndEmailSeriousProblemS113(Throwable ex, HttpServletRequest aRequest) {
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    // changed to allow compilation
    System.out.println("SERIOUS PROBLEM OCCURRED.");
    // changed to allow compilation
    System.out.println(troubleTicket.toString());
    aRequest
        .getSession()
        .getServletContext()
        .setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    // changed to allow compilation
    troubleTicket.toString();
  }
}
