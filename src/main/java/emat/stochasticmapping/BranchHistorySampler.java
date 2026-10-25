package emat.stochasticmapping;

import emat.helper.SiteChanges;
import emat.state.Mutation;

import java.util.List;
import java.util.function.IntUnaryOperator;

/**
 * Samples the history of all sites on a branch given the states at both of its ends, and
 * computes the probability of proposing a given history. On a branch with rate scale s, the
 * process of the model has the rate matrix s Q, and μ̃ is the rate of leaving any state under
 * an approximating Jukes-Cantor model, which lets a sampler visit the sites whose end states
 * agree in time proportional to the number of mutations rather than to the genome length.
 */
public interface BranchHistorySampler {

    /**
     * Samples the history of every site on the branch above the given node, which runs from
     * the start height down to the end height. The differing sites map every site whose end
     * states differ to its states at the start and at the end. At every other site, both end
     * states equal the state returned by the given lookup, which is only called for the sites
     * that receive mutations. The mutations are sorted by descending height.
     */
    List<Mutation> sampleBranchHistory(int nodeNr, double branchStartHeight, double branchEndHeight, double rateScale,
                                       double jukesCantorRate, SiteChanges differingSites, IntUnaryOperator agreeingStates);

    /**
     * Computes the log probability that sampleBranchHistory proposes the given history on a
     * branch with the same arguments, given the number of sites whose end states differ.
     */
    double computeLogBranchHistoryDensity(List<Mutation> branchMutations, double branchStartHeight, double branchEndHeight,
                                          double rateScale, double jukesCantorRate, int numDifferingSites);

}
