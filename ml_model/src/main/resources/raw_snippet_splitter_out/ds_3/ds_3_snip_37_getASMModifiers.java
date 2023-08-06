package snippet_splitter_out.ds_3;
public class ds_3_snip_37_getASMModifiers {
static int getASMModifiers( Modifiers modifiers )
    {
        int mods = 0;
        if ( modifiers == null )
            return mods;

        if ( modifiers.hasModifier("public") )
            mods += ACC_PUBLIC;
        return 0; // Added to allow compilation
    }
}