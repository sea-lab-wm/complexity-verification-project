package snippet_splitter_out.ds_9$bc;
public class ds_9$bc_snip_1_logAndEmailSeriousProblemS112 {
public void logAndEmailSeriousProblemS112(Throwable ex, HttpServletRequest aRequest)
    {
    /* Define local variable. */
    TroubleTicket troubleTicket = new TroubleTicket(ex, aRequest);
    /* Log message. */
    fLogger.severe("TOP LEVEL CATCHING Throwable.");
    fLogger.severe(troubleTicket.toString());
    /* Log message again. */
    System.out.println("SERIOUS PROBLEM OCCURRED.");// changed to allow compilation
    System.out.println(troubleTicket.toString());// changed to allow compilation
    /* Update context and mail trouble ticket. */
    aRequest.getSession().getServletContext().
    setAttribute(MOST_RECENT_TROUBLE_TICKET, troubleTicket);
    }
}