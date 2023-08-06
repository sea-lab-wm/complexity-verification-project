package snippet_splitter_out.ds_3;
public class ds_3_snip_21_swap {
public NameSpace swap( NameSpace newTop ) {
        NameSpace oldTop = (NameSpace)(stack.elementAt(0));
        stack.setElementAt( newTop, 0 );
        return oldTop;
    }
}