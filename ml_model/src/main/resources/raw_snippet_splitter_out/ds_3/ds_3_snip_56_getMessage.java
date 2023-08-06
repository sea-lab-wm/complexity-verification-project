package snippet_splitter_out.ds_3;
public class ds_3_snip_56_getMessage {
public String getMessage() {
        StringBuilder builder= new StringBuilder();
        if (fMessage != null)
            builder.append(fMessage);
        builder.append("arrays first differed at element ");

        return new String();                                                            /*Altered return*/
        //return null; // Added to allow compilation
    }
}