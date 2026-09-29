package emat.helper;

import beast.base.util.Randomizer;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.IntUnaryOperator;

/**
 * Samples the history of all sites on a branch given the states at both of its ends, under a
 * Jukes-Cantor model with a fictitious mutation rate μ̃ and no site-rate heterogeneity
 * (docs/mcmc-moves.md §3.4 and §4). As every site follows the same model, the sites whose
 * end states agree are mapped together in time roughly proportional to the number of sites
 * whose end states differ, rather than to the genome length. The proposal does not need to
 * match the real model, since the genetic prior corrects for it in the acceptance probability.
 * Single sites are mapped with the rate μ̃ = s μ̄ on a branch with rate scale s, where μ̄ is
 * the average escape rate of the rate matrix Q, which does not depend on the end states.
 */
public class JukesCantorStochasticMapping implements StochasticMapping {

    // safeguard against endless rejection sampling caused by numerical problems
    private static final int MAX_ATTEMPTS = 100000;

    // below this expected number of jumps, Poisson tails are summed directly to avoid cancellation
    private static final double SMALL_EXPECTED_JUMPS = 1.0;

    final int numStates;
    final int numSites;

    // the average escape rate μ̄ of the rate matrix Q set last
    double meanEscapeRate = Double.NaN;

    public JukesCantorStochasticMapping(int numStates, int numSites) {
        this.numStates = numStates;
        this.numSites = numSites;
    }

    /* Single Sites */

    @Override
    public void setRateMatrix(double[] rateMatrix) {
        double totalEscapeRate = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            totalEscapeRate -= rateMatrix[state * this.numStates + state];
        }
        this.meanEscapeRate = totalEscapeRate / this.numStates;
    }

    /**
     * Samples the history of the given site under Jukes-Cantor with the rate μ̃ = s μ̄, by
     * rejection sampling of the jump chain. The end probability is not needed, as the
     * Jukes-Cantor transition probabilities are known in closed form.
     */
    @Override
    public List<Mutation> sampleSiteHistory(int nodeNr, int site, double branchStartHeight, double branchEndHeight,
                                            int startState, int endState, double rateScale, double endProbability) {
        double duration = branchStartHeight - branchEndHeight;
        double expectedJumps = rateScale * this.meanEscapeRate * duration;

        List<Mutation> siteMutations = new ArrayList<>();
        if (duration <= 0.0 || expectedJumps <= 0.0) {
            // without time or rate, no mutations can occur and the end states are equal
            return siteMutations;
        }

        int minJumps = startState == endState ? 0 : 1;
        int[] states = this.sampleJumpStates(expectedJumps, minJumps, startState, endState);
        this.addSiteMutations(siteMutations, nodeNr, site, branchStartHeight, duration, states);

        return siteMutations;
    }

    /**
     * Computes the Jukes-Cantor density of the history with the rate μ̃ = s μ̄,
     * e^{-μ̃ t} (μ̃ / (K - 1))^M for M mutations and K states, divided by the transition
     * probability from the start to the end state.
     */
    @Override
    public double computeLogSiteHistoryDensity(List<Mutation> siteMutations, double branchStartHeight, double branchEndHeight,
                                               int startState, int endState, double rateScale, double endProbability) {
        double duration = branchStartHeight - branchEndHeight;
        double mutationRate = rateScale * this.meanEscapeRate;
        double expectedJumps = mutationRate * duration;

        if (duration <= 0.0 || expectedJumps <= 0.0) {
            return 0.0;
        }

        double logDensity = -expectedJumps + siteMutations.size() * Math.log(mutationRate / (this.numStates - 1.0));

        double changeProbability = this.computeChangeProbability(expectedJumps);
        logDensity -= startState == endState
                ? Math.log1p(-(this.numStates - 1.0) * changeProbability)
                : Math.log(changeProbability);

        return logDensity;
    }

    /* Branch Histories */

    /**
     * Samples the history of every site on the branch above the given node, which runs from
     * the start height down to the end height, under Jukes-Cantor with the given rate μ̃ of
     * leaving any state. The differing sites map every site whose end states differ to its
     * states at the start and at the end. At every other site, both end states equal the
     * state returned by the given lookup, which is only called for the rare sites that
     * receive mutations. The mutations are sorted by descending height.
     */
    public List<Mutation> sampleBranchHistory(int nodeNr, double branchStartHeight, double branchEndHeight, double mutationRate,
                                              SiteChanges differingSites, IntUnaryOperator agreeingStates) {
        // evolution runs forwards in time from the branch start to the branch end

        double duration = branchStartHeight - branchEndHeight;

        List<Mutation> branchMutations = new ArrayList<>();
        if (duration <= 0.0) {
            // a branch of length zero cannot carry mutations, and its end states are equal
            return branchMutations;
        }

        double expectedJumps = mutationRate * duration;

        // map the sites whose end states differ one by one, as they need at least one jump

        for (int slot = 0; slot < differingSites.getSize(); slot++) {
            int startState = differingSites.getStartState(slot);
            int endState = differingSites.getEndState(slot);

            int[] states = this.sampleJumpStates(expectedJumps, 1, startState, endState);
            this.addSiteMutations(branchMutations, nodeNr, differingSites.getSite(slot), branchStartHeight, duration, states);
        }

        // map all sites whose end states agree at once (Algorithm 1)

        this.sampleAgreeingSiteHistories(branchMutations, nodeNr, branchStartHeight, duration, expectedJumps,
                differingSites, agreeingStates);

        branchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        return branchMutations;
    }

    /**
     * Computes the log probability α_mut that sampleBranchHistory proposes the given history
     * on a branch, given the number of sites whose end states differ. This is the
     * Jukes-Cantor density of the history, e^{-μ̃ L t} (μ̃ / (K - 1))^M for M mutations and K
     * states, divided by the transition probabilities of all sites.
     */
    public double computeLogBranchHistoryDensity(List<Mutation> branchMutations, double branchStartHeight, double branchEndHeight,
                                                 double mutationRate, int numDifferingSites) {
        double duration = branchStartHeight - branchEndHeight;
        double expectedJumps = mutationRate * duration;

        // the transition probabilities of Jukes-Cantor for differing and agreeing end states

        double changeProbability = this.computeChangeProbability(expectedJumps);
        double logChangeProbability = Math.log(changeProbability);
        double logStayProbability = Math.log1p(-(this.numStates - 1.0) * changeProbability);

        double logDensity = -expectedJumps * this.numSites
                + branchMutations.size() * Math.log(mutationRate / (this.numStates - 1.0));

        logDensity -= numDifferingSites * logChangeProbability
                + (this.numSites - numDifferingSites) * logStayProbability;

        return logDensity;
    }

    /* Agreeing Sites */

    /**
     * Maps all sites whose end states agree. Per site, Nielsen's rejection sampler proposes
     * an empty history with probability p0 = e^{-μ̃ t}, a single jump that is always rejected
     * with probability p1 = μ̃ t e^{-μ̃ t}, and two or more jumps otherwise. It thus ends with
     * an empty history before proposing two or more jumps with probability p* = p0 / (1 - p1).
     * The number of sites that trivially stay empty is drawn from Geom(1 - p*), and only the
     * site after them is sampled in detail. On rejection, the sampling restarts at that site.
     */
    private void sampleAgreeingSiteHistories(List<Mutation> branchMutations, int nodeNr, double branchStartHeight, double duration,
                                             double expectedJumps, SiteChanges differingSites, IntUnaryOperator agreeingStates) {
        // 1 - p* = P(n ≥ 2) / (1 - p1), computed without cancellation for short branches

        double multipleJumpsProbability = this.computePoissonTailProbability(expectedJumps, 2);
        double detailedProbability = multipleJumpsProbability / (1.0 - expectedJumps * Math.exp(-expectedJumps));

        if (detailedProbability <= 0.0) {
            return;
        }

        double site = 0;
        int previousSiteIndex = -1;
        int attempts = 0;

        while (true) {
            site += this.sampleGeometric(detailedProbability);
            if (site >= this.numSites) {
                return;
            }

            int siteIndex = (int) site;

            if (differingSites.containsSite(siteIndex)) {
                // the differing sites are mapped separately, which is equivalent to filtering them out here
                site++;
                continue;
            }

            // count the rejections at the current site only
            attempts = siteIndex == previousSiteIndex ? attempts + 1 : 1;
            previousSiteIndex = siteIndex;
            if (attempts > MAX_ATTEMPTS) {
                throw new RuntimeException("Could not sample the history of the agreeing sites on a branch.");
            }

            // propose two or more jumps, whose states only depend on the start state by symmetry

            int numJumps = this.sampleTruncatedPoisson(expectedJumps, 2);
            int[] states = new int[numJumps + 1];
            for (int i = 1; i <= numJumps; i++) {
                states[i] = this.sampleOtherState(states[i - 1]);
            }

            if (states[numJumps] != states[0]) {
                // restart the rejection sampling at the same site
                continue;
            }

            // rotate the states cyclically so that the history starts and ends in the actual state

            int state = agreeingStates.applyAsInt(siteIndex);
            for (int i = 0; i <= numJumps; i++) {
                states[i] = (states[i] + state) % this.numStates;
            }

            this.addSiteMutations(branchMutations, nodeNr, siteIndex, branchStartHeight, duration, states);
            site++;
        }
    }

    /* Single Sites */

    /**
     * Samples the states of a Jukes-Cantor jump chain from the start to the end state by
     * rejection: the number of jumps is drawn from a Poisson distribution restricted to at
     * least the given minimum, and every jump moves to a uniformly chosen other state. The
     * returned array holds the start state followed by the state after every jump.
     */
    private int[] sampleJumpStates(double expectedJumps, int minJumps, int startState, int endState) {
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            int numJumps = this.sampleTruncatedPoisson(expectedJumps, minJumps);

            int[] states = new int[numJumps + 1];
            states[0] = startState;
            for (int i = 1; i <= numJumps; i++) {
                states[i] = this.sampleOtherState(states[i - 1]);
            }

            if (states[numJumps] == endState) {
                return states;
            }
        }

        throw new RuntimeException("Could not sample the history of a site on a branch.");
    }

    /**
     * Adds the mutations of the given jump chain at the given site, with jump times drawn
     * uniformly on the branch.
     */
    private void addSiteMutations(List<Mutation> branchMutations, int nodeNr, int site, double branchStartHeight,
                                  double duration, int[] states) {
        int numJumps = states.length - 1;

        double[] jumpTimes = new double[numJumps];
        for (int i = 0; i < numJumps; i++) {
            jumpTimes[i] = Randomizer.nextDouble() * duration;
        }
        Arrays.sort(jumpTimes);

        double previousHeight = branchStartHeight;
        for (int i = 0; i < numJumps; i++) {
            double height = branchStartHeight - jumpTimes[i];
            branchMutations.add(new Mutation(nodeNr, height, previousHeight, site, states[i], states[i + 1]));
            previousHeight = height;
        }
    }

    /**
     * Computes the Jukes-Cantor probability of ending in a given other state after the given
     * expected number of jumps.
     */
    private double computeChangeProbability(double expectedJumps) {
        return -Math.expm1(-expectedJumps * this.numStates / (this.numStates - 1.0)) / this.numStates;
    }

    /* Random Draws */

    /** Samples a state uniformly among all states other than the given one. */
    private int sampleOtherState(int state) {
        int otherState = Randomizer.nextInt(this.numStates - 1);
        return otherState >= state ? otherState + 1 : otherState;
    }

    /**
     * Samples the number of failures before the first success of Bernoulli trials with the
     * given success probability. The result may exceed the range of int for tiny
     * probabilities, so it is returned as a double.
     */
    private double sampleGeometric(double successProbability) {
        if (successProbability >= 1.0) {
            return 0.0;
        }
        return Math.floor(Math.log(Randomizer.nextDouble()) / Math.log1p(-successProbability));
    }

    /**
     * Samples from a Poisson distribution with the given mean, restricted to at least the
     * given minimum. For small means, this inverts the restricted distribution directly,
     * with every weight taken relative to the weight of the minimum to avoid underflow.
     */
    private int sampleTruncatedPoisson(double mean, int minimum) {
        if (mean >= SMALL_EXPECTED_JUMPS) {
            for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
                long count = Randomizer.nextPoisson(mean);
                if (count >= minimum) {
                    return (int) count;
                }
            }
            throw new RuntimeException("Could not sample from a truncated Poisson distribution.");
        }

        double totalWeight = 0.0;
        double weight = 1.0;
        for (int count = minimum; weight > Math.ulp(totalWeight); count++) {
            totalWeight += weight;
            weight *= mean / (count + 1);
        }

        double threshold = Randomizer.nextDouble() * totalWeight;
        double cumulativeWeight = 0.0;
        weight = 1.0;
        int count = minimum;

        while (true) {
            cumulativeWeight += weight;
            if (threshold < cumulativeWeight || weight == 0.0) {
                return count;
            }
            weight *= mean / (count + 1);
            count++;
        }
    }

    /**
     * Computes the probability that a Poisson distribution with the given mean is at least
     * the given minimum. For small means, the tail is summed directly, as subtracting the
     * complement from one would lose all precision.
     */
    private double computePoissonTailProbability(double mean, int minimum) {
        if (mean >= SMALL_EXPECTED_JUMPS) {
            double complement = 0.0;
            double term = Math.exp(-mean);
            for (int count = 0; count < minimum; count++) {
                complement += term;
                term *= mean / (count + 1);
            }
            return 1.0 - complement;
        }

        double term = Math.exp(-mean);
        for (int count = 1; count <= minimum; count++) {
            term *= mean / count;
        }

        double tail = 0.0;
        for (int count = minimum + 1; term > Math.ulp(tail); count++) {
            tail += term;
            term *= mean / count;
        }
        return tail;
    }

}
