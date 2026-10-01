package emat.stochasticmapping;

import beast.base.util.Randomizer;
import emat.helper.SiteChanges;
import emat.helper.SiteStates;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Comparator;
import java.util.List;
import java.util.function.IntPredicate;
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
        return logDensity - this.computeLogEndStatesProbability(expectedJumps, 1, startState == endState ? 0 : 1);
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

        double logDensity = -expectedJumps * this.numSites
                + branchMutations.size() * Math.log(mutationRate / (this.numStates - 1.0));
        return logDensity - this.computeLogEndStatesProbability(expectedJumps, this.numSites, numDifferingSites);
    }

    /* Star Centres */

    /**
     * The Jukes-Cantor model of a star of branches that meet at a centre: per branch, the
     * probability of ending in a given other state, and, for a site where all outer ends
     * agree, the log probabilities of the centre keeping their state or taking a given other
     * state, and the probability of it taking any other state.
     */
    private record StarModel(double[] changeProbabilities, double logSameProbability, double logOtherProbability,
                             double centreChangeProbability) {
    }

    /**
     * Samples the states at the centre of a star of branches given the states at their
     * outer ends, under Jukes-Cantor with the given expected number of jumps per branch.
     * Jukes-Cantor is time-reversible, so the direction of the branches does not matter.
     * The star states hold the outer ends in the columns below the number of branches and
     * the centre in the column of that number, and initially hold every site where the
     * states of the outer ends do not all agree. At every other site, all outer ends are in
     * the state returned by the given lookup, and the centre only differs from it if it
     * differs at all branches, which is rare. Such sites are found by skipping sites
     * geometrically as in Algorithm 1 and added to the star states.
     * <p>
     * At the given free sites, which are sorted, the outer end in the free column is not
     * fixed, e.g. a tip with missing data. The centre is then sampled from the other outer
     * ends only, and the free end afterwards from the centre, which also adds every free site
     * where it differs from the centre. Returns the log probability of all sampled states.
     */
    public double sampleStarStates(double[] expectedJumps, SiteStates starStates, IntUnaryOperator agreeingStates,
                                   int[] freeSites, BitSet freeSiteSet, int freeColumn) {
        int centre = expectedJumps.length;
        StarModel fullModel = this.createStarModel(expectedJumps, -1);
        StarModel freeModel = this.createStarModel(expectedJumps, freeColumn);
        double[] weights = new double[this.numStates];

        // sample the centre at the sites held initially, whose outer ends mostly differ

        int numInitialSites = starStates.getSize();
        int numInitialFreeSites = 0;
        double logProbability = 0.0;

        for (int slot = 0; slot < numInitialSites; slot++) {
            boolean isFree = freeSiteSet.get(starStates.getSite(slot));
            numInitialFreeSites += isFree ? 1 : 0;

            double totalWeight = this.computeCentreWeights(isFree ? freeModel : fullModel, starStates, slot, isFree ? freeColumn : -1, weights);
            int state = StochasticMapping.sampleIndex(weights);
            starStates.setState(slot, centre, state);
            logProbability += Math.log(weights[state] / totalWeight);
        }

        // sample the centre at the other sites, where the outer ends agree

        int numChangedSites = this.sampleChangedCentres(fullModel, starStates, agreeingStates, null, freeSiteSet);
        logProbability += this.computeLogAgreeingCentreProbability(
                fullModel, this.numSites - freeSites.length - (numInitialSites - numInitialFreeSites), numChangedSites
        );

        int numChangedFreeSites = this.sampleChangedCentres(freeModel, starStates, agreeingStates, freeSites, null);
        logProbability += this.computeLogAgreeingCentreProbability(
                freeModel, freeSites.length - numInitialFreeSites, numChangedFreeSites
        );

        if (freeSites.length == 0) {
            return logProbability;
        }

        // sample the free end from the centre, explicitly at the held sites and by skipping at the others

        double changeProbability = (this.numStates - 1.0) * this.computeChangeProbability(expectedJumps[freeColumn]);
        int numDifferingFreeSites = 0;

        for (int slot = 0; slot < starStates.getSize(); slot++) {
            if (!freeSiteSet.get(starStates.getSite(slot))) {
                continue;
            }

            int centreState = starStates.getState(slot, centre);
            int state = Randomizer.nextDouble() < changeProbability ? this.sampleOtherState(centreState) : centreState;
            starStates.setState(slot, freeColumn, state);
            numDifferingFreeSites += state != centreState ? 1 : 0;
        }

        int numHeldSites = starStates.getSize();
        numDifferingFreeSites += this.visitRandomIndices(freeSites.length, changeProbability, freeSiteNr -> {
            int site = freeSites[freeSiteNr];

            // the held sites were sampled explicitly, including the ones added here, whose positions only increase
            int heldSlot = starStates.getSlot(site);
            if (heldSlot >= 0 && heldSlot < numHeldSites) {
                return false;
            }

            int agreeingState = agreeingStates.applyAsInt(site);
            int slot = starStates.addSite(site);
            starStates.setAllStates(slot, agreeingState);
            starStates.setState(slot, freeColumn, this.sampleOtherState(agreeingState));
            return true;
        });

        return logProbability + this.computeLogEndStatesProbability(expectedJumps[freeColumn], freeSites.length, numDifferingFreeSites);
    }

    /**
     * Computes the log probability that sampleStarStates samples the centre and free states
     * held by the given star states. Every site without an entry keeps the state of the
     * outer ends at the centre and at the free end.
     */
    public double computeLogStarStatesDensity(double[] expectedJumps, SiteStates starStates, int numFreeSites,
                                              BitSet freeSiteSet, int freeColumn) {
        int centre = expectedJumps.length;
        StarModel fullModel = this.createStarModel(expectedJumps, -1);
        StarModel freeModel = this.createStarModel(expectedJumps, freeColumn);
        double[] weights = new double[this.numStates];

        int[] numOuterSites = new int[2];
        int[] numChangedSites = new int[2];
        int numDifferingFreeSites = 0;
        double logProbability = 0.0;

        for (int slot = 0; slot < starStates.getSize(); slot++) {
            boolean isFree = freeSiteSet.get(starStates.getSite(slot));
            int modelNr = isFree ? 1 : 0;
            int ignoredColumn = isFree ? freeColumn : -1;
            int centreState = starStates.getState(slot, centre);

            if (isFree && starStates.getState(slot, freeColumn) != centreState) {
                numDifferingFreeSites++;
            }

            int agreeingState = this.getAgreeingState(starStates, slot, centre, ignoredColumn);
            if (agreeingState >= 0) {
                numChangedSites[modelNr] += centreState != agreeingState ? 1 : 0;
                continue;
            }

            numOuterSites[modelNr]++;
            double totalWeight = this.computeCentreWeights(isFree ? freeModel : fullModel, starStates, slot, ignoredColumn, weights);
            logProbability += Math.log(weights[centreState] / totalWeight);
        }

        logProbability += this.computeLogAgreeingCentreProbability(
                fullModel, this.numSites - numFreeSites - numOuterSites[0], numChangedSites[0]
        );
        logProbability += this.computeLogAgreeingCentreProbability(
                freeModel, numFreeSites - numOuterSites[1], numChangedSites[1]
        );

        if (numFreeSites == 0) {
            return logProbability;
        }

        return logProbability + this.computeLogEndStatesProbability(expectedJumps[freeColumn], numFreeSites, numDifferingFreeSites);
    }

    /**
     * Finds the sites where the outer ends agree but the centre differs, by skipping sites
     * geometrically, and adds them to the star states. The candidates are either all sites
     * except the given excluded ones, or only the given sites. Sites already held are
     * skipped, which is equivalent to filtering them out. Returns the number of added sites.
     */
    private int sampleChangedCentres(StarModel model, SiteStates starStates, IntUnaryOperator agreeingStates,
                                     int[] candidateSites, BitSet excludedSites) {
        int centre = starStates.getNumColumns() - 1;
        int numCandidates = candidateSites == null ? this.numSites : candidateSites.length;

        return this.visitRandomIndices(numCandidates, model.centreChangeProbability(), candidateNr -> {
            int site = candidateSites == null ? candidateNr : candidateSites[candidateNr];
            if (starStates.containsSite(site) || (excludedSites != null && excludedSites.get(site))) {
                return false;
            }

            int agreeingState = agreeingStates.applyAsInt(site);
            int slot = starStates.addSite(site);
            starStates.setAllStates(slot, agreeingState);
            starStates.setState(slot, centre, this.sampleOtherState(agreeingState));
            return true;
        });
    }

    /**
     * Computes the model of a star with the given expected number of jumps per branch,
     * leaving out the branch of the given column unless it is negative. At a site where all
     * outer ends agree, the centre keeps their state with a weight of ∏ P=(i) and takes a
     * given other state with a weight of ∏ P≠(i). The ratio r of the latter to the former is
     * tiny on short branches, so the probabilities are computed from it without
     * cancellation.
     */
    private StarModel createStarModel(double[] expectedJumps, int ignoredColumn) {
        double[] changeProbabilities = new double[expectedJumps.length];
        double logStayProduct = 0.0;
        double logChangeProduct = 0.0;

        for (int i = 0; i < expectedJumps.length; i++) {
            changeProbabilities[i] = this.computeChangeProbability(expectedJumps[i]);
            if (i != ignoredColumn) {
                logStayProduct += Math.log1p(-(this.numStates - 1.0) * changeProbabilities[i]);
                logChangeProduct += Math.log(changeProbabilities[i]);
            }
        }

        double changeRatio = Math.exp(logChangeProduct - logStayProduct);
        double logSameProbability = -Math.log1p((this.numStates - 1.0) * changeRatio);
        double logOtherProbability = logChangeProduct - logStayProduct + logSameProbability;
        double centreChangeProbability = (this.numStates - 1.0) * changeRatio / (1.0 + (this.numStates - 1.0) * changeRatio);

        return new StarModel(changeProbabilities, logSameProbability, logOtherProbability, centreChangeProbability);
    }

    /**
     * Computes the weight of every centre state at the given slot, the product of the
     * transition probabilities from it to the outer ends except the one in the given column,
     * and returns their sum.
     */
    private double computeCentreWeights(StarModel model, SiteStates starStates, int slot, int ignoredColumn, double[] weights) {
        double[] changeProbabilities = model.changeProbabilities();
        double totalWeight = 0.0;

        for (int state = 0; state < this.numStates; state++) {
            double weight = 1.0;
            for (int i = 0; i < changeProbabilities.length; i++) {
                if (i == ignoredColumn) {
                    continue;
                }
                weight *= starStates.getState(slot, i) == state
                        ? 1.0 - (this.numStates - 1.0) * changeProbabilities[i]
                        : changeProbabilities[i];
            }
            weights[state] = weight;
            totalWeight += weight;
        }

        return totalWeight;
    }

    /**
     * Returns the state of the outer ends at the given slot except the one in the given
     * column, or -1 if they do not all agree.
     */
    private int getAgreeingState(SiteStates starStates, int slot, int centre, int ignoredColumn) {
        int agreeingState = -1;
        for (int i = 0; i < centre; i++) {
            if (i == ignoredColumn) {
                continue;
            }
            int state = starStates.getState(slot, i);
            if (agreeingState < 0) {
                agreeingState = state;
            } else if (state != agreeingState) {
                return -1;
            }
        }
        return agreeingState;
    }

    /**
     * Computes the log probability of the centre states at the given number of sites where
     * the outer ends agree, of which the given number have a centre in another state.
     */
    private double computeLogAgreeingCentreProbability(StarModel model, int numAgreeingSites, int numChangedSites) {
        return this.computeLogCountProbability(
                numAgreeingSites, numChangedSites, model.logSameProbability(), model.logOtherProbability()
        );
    }

    /* Free End States */

    /**
     * Samples the end states of a branch at the given free sites, which are sorted, from
     * their start states under Jukes-Cantor with the given expected number of jumps. Every
     * free site whose end state differs from its start state is added to the given
     * differing sites, which must not hold any free site yet. Only these sites call the
     * lookup of the start states. Returns the number of added sites.
     */
    public int sampleFreeEndStates(double expectedJumps, int[] freeSites, IntUnaryOperator startStates, SiteChanges differingSites) {
        double changeProbability = (this.numStates - 1.0) * this.computeChangeProbability(expectedJumps);

        return this.visitRandomIndices(freeSites.length, changeProbability, freeSiteNr -> {
            int site = freeSites[freeSiteNr];
            int startState = startStates.applyAsInt(site);
            differingSites.addSite(site, startState, this.sampleOtherState(startState));
            return true;
        });
    }

    /**
     * Computes the log probability that sampleFreeEndStates samples end states that differ
     * from their start states at the given number of the free sites.
     */
    public double computeLogFreeEndStatesDensity(double expectedJumps, int numFreeSites, int numDifferingFreeSites) {
        return this.computeLogEndStatesProbability(expectedJumps, numFreeSites, numDifferingFreeSites);
    }

    /* Probabilities */

    /**
     * Computes the Jukes-Cantor probability of ending in a given other state after the given
     * expected number of jumps.
     */
    private double computeChangeProbability(double expectedJumps) {
        return -Math.expm1(-expectedJumps * this.numStates / (this.numStates - 1.0)) / this.numStates;
    }

    /**
     * Computes the Jukes-Cantor log probability of the end states of the given number of
     * sites after the given expected number of jumps, where the given number of them differ
     * from their start states.
     */
    private double computeLogEndStatesProbability(double expectedJumps, int numSites, int numDifferingSites) {
        double changeProbability = this.computeChangeProbability(expectedJumps);
        return this.computeLogCountProbability(
                numSites, numDifferingSites, Math.log1p(-(this.numStates - 1.0) * changeProbability), Math.log(changeProbability)
        );
    }

    /**
     * Computes the log probability of the given number of sites, of which the given number
     * take a given other state and the rest keep their state, from the log probabilities of
     * both outcomes at a single site.
     */
    private double computeLogCountProbability(int numSites, int numChangedSites, double logSameProbability,
                                              double logOtherProbability) {
        double logProbability = (numSites - numChangedSites) * logSameProbability;

        // avoid 0 · -∞ if no site can change its state
        if (numChangedSites > 0) {
            logProbability += numChangedSites * logOtherProbability;
        }

        return logProbability;
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
            int[] states = this.sampleJumpChain(numJumps, 0);
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

    /* Jump Chains */

    /**
     * Samples the states of a Jukes-Cantor jump chain from the start to the end state by
     * rejection: the number of jumps is drawn from a Poisson distribution restricted to at
     * least the given minimum, and every jump moves to a uniformly chosen other state. The
     * returned array holds the start state followed by the state after every jump.
     */
    private int[] sampleJumpStates(double expectedJumps, int minJumps, int startState, int endState) {
        for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
            int numJumps = this.sampleTruncatedPoisson(expectedJumps, minJumps);
            int[] states = this.sampleJumpChain(numJumps, startState);
            if (states[numJumps] == endState) {
                return states;
            }
        }

        throw new RuntimeException("Could not sample the history of a site on a branch.");
    }

    /**
     * Samples a Jukes-Cantor jump chain with the given number of jumps from the given start
     * state, where every jump moves to a uniformly chosen other state. The returned array
     * holds the start state followed by the state after every jump.
     */
    private int[] sampleJumpChain(int numJumps, int startState) {
        int[] states = new int[numJumps + 1];
        states[0] = startState;
        for (int i = 1; i <= numJumps; i++) {
            states[i] = this.sampleOtherState(states[i - 1]);
        }
        return states;
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

    /* Random Draws */

    /**
     * Selects every index below the given number independently with the given probability,
     * by skipping indices geometrically, and passes it to the given action. Returns the number
     * of selected indices that the action accepts.
     */
    private int visitRandomIndices(int numIndices, double probability, IntPredicate action) {
        int numAccepted = 0;
        double index = 0;

        while (probability > 0.0) {
            index += this.sampleGeometric(probability);
            if (index >= numIndices) {
                break;
            }

            if (action.test((int) index)) {
                numAccepted++;
            }
            index++;
        }

        return numAccepted;
    }

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
