package snippet_splitter_out.ds_3;
public class ds_3_snip_12_getGlobal {
public This getGlobal( Interpreter declaringInterpreter )
    {
        if ( parent != null )
            return parent.getGlobal( declaringInterpreter );
        else
            return getThis( declaringInterpreter );
    }
}