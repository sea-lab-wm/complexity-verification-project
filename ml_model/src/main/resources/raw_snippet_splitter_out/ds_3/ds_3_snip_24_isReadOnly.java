package snippet_splitter_out.ds_3;
public class ds_3_snip_24_isReadOnly {
public boolean isReadOnly() throws HsqlException {

        Object info = getAttribute(Session.INFO_CONNECTION_READONLY);

        isReadOnly = ((Boolean) info).booleanValue();

        return isReadOnly;
    }
}