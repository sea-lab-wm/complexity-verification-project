package snippet_splitter_out.ds_3;
public class ds_3_snip_91_printFailures {
protected void printFailures(Result result) {
        if (result.getFailureCount() == 0)
            return;
        if (result.getFailureCount() == 1)
            getWriter().println("There was " + result.getFailureCount() + " failure:");
        else
            getWriter().println("There were " + result.getFailureCount() + " failures:");
    }
}