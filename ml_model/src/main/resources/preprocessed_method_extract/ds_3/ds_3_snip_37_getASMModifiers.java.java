// Snippet s71
/**
 * Translate bsh.Modifiers into ASM modifier bitflags.
 */
// SNIPPET_STARTS
static int getASMModifiers(Modifiers modifiers) {
    int mods = 0;
    if (modifiers == null)
        return mods;
    if (modifiers.hasModifier("public"))
        mods += ACC_PUBLIC;
    // Added to allow compilation
    return 0;
}