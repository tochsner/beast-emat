package emat.stochasticmapping;

/**
 * The samplers of the history on whole branches: JUKES_CANTOR samples every path under the
 * approximating Jukes-Cantor model, and HYBRID chooses the sites with mutations with one
 * probability for all states, derived from the model of the genetic prior, and then samples
 * their paths exactly under that model.
 */
public enum HistorySampler {
    JUKES_CANTOR, HYBRID;

    /** Creates the sampler, which reads the model from the given process. */
    public BranchHistorySampler create(SubstitutionProcess process, int numSites) {
        return switch (this) {
            case JUKES_CANTOR -> new JukesCantorStochasticMapping(process.getNumStates(), numSites);
            case HYBRID -> new HybridBranchHistories(process, numSites);
        };
    }

}
