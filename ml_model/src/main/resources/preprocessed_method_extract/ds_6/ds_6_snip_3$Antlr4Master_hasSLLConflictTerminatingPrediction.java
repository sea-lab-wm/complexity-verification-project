// org.antlr.v4.runtime.atn.PredictionMode.hasSLLConflictTerminatingPrediction(org.antlr.v4.runtime.atn.PredictionMode,org.antlr.v4.runtime.atn.ATNConfigSet)
// SNIPPET_STARTS
public static boolean hasSLLConflictTerminatingPrediction(PredictionMode mode, ATNConfigSet configs) {
    /* Configs in rule stop states indicate reaching the end of the decision
     * rule (local context) or end of start rule (full context). If all
     * configs meet this condition, then none of the configurations is able
     * to match additional input so we terminate prediction.
     */
    if (allConfigsInRuleStopStates(configs)) {
        return true;
    }
    // pure SLL mode parsing
    if (mode == PredictionMode.SLL) {
        // Don't bother with combining configs from different semantic
        // contexts if we can fail over to full LL; costs more time
        // since we'll often fail over anyway.
        if (configs.hasSemanticContext) {
            // dup configs, tossing out semantic predicates
            ATNConfigSet dup = new ATNConfigSet();
            for (ATNConfig c : configs) {
                c = new ATNConfig(c, SemanticContext.NONE);
                dup.add(c);
            }
            configs = dup;
        }
        // now we have combined contexts for configs with dissimilar preds
    }
    // pure SLL or combined SLL+LL mode parsing
    Collection<BitSet> altsets = getConflictingAltSubsets(configs);
    boolean heuristic = hasConflictingAltSet(altsets) && !hasStateAssociatedWithOneAlt(configs);
    return heuristic;
}