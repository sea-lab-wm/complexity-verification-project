package snippet_splitter_out.ds_f;
public class ds_f_snip_YesNo_compute {
// SNIPPET_STARTS
static Boolean compute(String input) {
    input = input.toLowerCase();
    if (input.contentEquals("n")) {
        return false;
    } else if (input.contentEquals("no")) {
        return false;
    }
    if (input.contentEquals("y")) {
        return true;
    } else if (input.contentEquals("yes")) {
        return true;
    }
    return null;
}
}