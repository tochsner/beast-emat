package esat.stochasticmapping;

import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import esat.helper.TipCompatibility;
import esat.state.NodeStates;
import esat.state.StateChanges;

import java.util.Arrays;

/**
 * Samples the states of a single node from their exact full conditional given the states of
 * its neighbours, i.e. its parent and its children. The sites are independent given the
 * neighbours, and the conditional of state x at a site is proportional to
 * <pre>
 *     w(x) = U(p, x) · Π_c P_c(x, c) · mask(x),
 * </pre>
 * where U is the transition probability matrix of the branch above the node, or the root
 * frequencies in every row for the root, P_c that of the branch above child c, and the mask
 * restricts the latent states of a tip to those compatible with its data.
 * <p>
 * Sites at which the neighbours disagree are found by merging their change lists and drawn
 * exactly. At a site at which every neighbour is in the same state a, the conditional only
 * depends on a, and leaves a with probability ε_a. Such sites are visited with the bound
 * r = max_a ε_a by geometric skipping, and a visited site leaves a with probability ε_a / r.
 * This thinning samples the exact conditional, so the cost scales with the number of changes
 * instead of the number of sites, without an approximation.
 * <p>
 * The sampler reads the model from the substitution process, which has to be updated before.
 * The sampled states are returned as changes relative to the current reference sequence, also
 * for the root.
 */
public class NodeStateSampler {

    private final NodeStates nodeStates;
    private final SubstitutionProcess substitutionProcess;
    private final TipCompatibility tipCompatibility;

    private final int numStates;
    private final int numSites;

    // the transition probabilities of the branch above the node, or the root frequencies in every row, and those of the branches above its children
    private final double[] upperProbabilities;
    private double[][] childProbabilities = new double[0][];

    // per state a in which all neighbours agree: the weights w(x) of the other states x and their sum, the weight of a, the total and the probability ε_a to leave a
    private final double[] leaveWeights;
    private final double[] leaveTotals;
    private final double[] stayWeights;
    private final double[] totalWeights;
    private final double[] leaveProbabilities;

    // the bound r on the probabilities to leave, and log(1 - r) for the geometric skipping
    private double maxLeaveProbability;
    private double logSkipProbability;

    // scratch space for the neighbour lists, the weights of a single site and the sampled changes
    private StateChanges[] neighbourChanges = new StateChanges[0];
    private final ChangeListMerge merge = new ChangeListMerge();
    private final double[] siteWeights;
    private final int[] referenceStateCounts;
    private final int[] sampledSites;
    private final int[] sampledStates;
    private int numSampledSites;

    public NodeStateSampler(NodeStates nodeStates, SubstitutionProcess substitutionProcess) {
        this.nodeStates = nodeStates;
        this.substitutionProcess = substitutionProcess;
        this.tipCompatibility = substitutionProcess.getTipCompatibility();

        this.numStates = nodeStates.getNumStates();
        this.numSites = nodeStates.getNumSites();
        if (substitutionProcess.getNumStates() != this.numStates) {
            throw new IllegalArgumentException("The node states have a different number of states than the substitution process.");
        }

        this.upperProbabilities = new double[this.numStates * this.numStates];
        this.leaveWeights = new double[this.numStates * this.numStates];
        this.leaveTotals = new double[this.numStates];
        this.stayWeights = new double[this.numStates];
        this.totalWeights = new double[this.numStates];
        this.leaveProbabilities = new double[this.numStates];

        this.siteWeights = new double[this.numStates];
        this.referenceStateCounts = new int[this.numStates];
        this.sampledSites = new int[this.numSites];
        this.sampledStates = new int[this.numSites];
    }

    /* Sampling */

    /**
     * Samples the states of the given node from their full conditional given its neighbours
     * and returns them as changes relative to the current reference sequence. They are applied
     * by NodeStates.applyRootStates for the root and by NodeStates.applyChanges otherwise.
     * The observed sites of a tip keep their current states, only its latent sites are sampled.
     */
    public StateChanges sample(Node node) {
        this.prepareNode(node);
        return node.isLeaf() ? this.sampleTip(node) : this.sampleInnerNode(node);
    }

    /**
     * Fetches the transition probabilities around the given node and computes, for every state
     * a in which the neighbours may agree, the weights w^(a)(x) = U(a, x) · Π_c P_c(x, a), the
     * probability ε_a to leave a and their maximum.
     */
    private void prepareNode(Node node) {
        int numChildren = node.getChildCount();
        if (this.childProbabilities.length < numChildren) {
            this.childProbabilities = new double[numChildren][this.numStates * this.numStates];
        }

        if (node.isRoot()) {
            double[] frequencies = this.substitutionProcess.getFrequencies();
            for (int from = 0; from < this.numStates; from++) {
                System.arraycopy(frequencies, 0, this.upperProbabilities, from * this.numStates, this.numStates);
            }
        } else {
            this.substitutionProcess.getTransitionProbabilities(node, this.upperProbabilities);
        }

        for (int c = 0; c < numChildren; c++) {
            this.substitutionProcess.getTransitionProbabilities(node.getChild(c), this.childProbabilities[c]);
        }

        // the probability to leave is computed from the weights of the other states, which stays accurate if it is small

        this.maxLeaveProbability = 0.0;
        for (int agreed = 0; agreed < this.numStates; agreed++) {
            double leaveWeight = 0.0;
            for (int state = 0; state < this.numStates; state++) {
                double weight = this.upperProbabilities[agreed * this.numStates + state];
                for (int c = 0; c < numChildren; c++) {
                    weight *= this.childProbabilities[c][state * this.numStates + agreed];
                }

                if (state == agreed) {
                    this.stayWeights[agreed] = weight;
                    this.leaveWeights[agreed * this.numStates + state] = 0.0;
                } else {
                    this.leaveWeights[agreed * this.numStates + state] = weight;
                    leaveWeight += weight;
                }
            }

            this.leaveTotals[agreed] = leaveWeight;
            this.totalWeights[agreed] = this.stayWeights[agreed] + leaveWeight;
            if (!(this.totalWeights[agreed] > 0.0)) {
                throw new IllegalStateException("The neighbours of node " + node.getNr() + " cannot agree in state " + agreed + ".");
            }
            this.leaveProbabilities[agreed] = leaveWeight / this.totalWeights[agreed];
            this.maxLeaveProbability = Math.max(this.maxLeaveProbability, this.leaveProbabilities[agreed]);
        }

        this.logSkipProbability = Math.log1p(-this.maxLeaveProbability);
    }

    /**
     * Samples an inner node or the root in one ascending pass over two streams of sites: the
     * sites at which some neighbour differs from the reference, from merging their changes,
     * and the sites visited by the thinning.
     */
    private StateChanges sampleInnerNode(Node node) {
        int[] reference = this.nodeStates.getReferenceSequence();
        boolean hasParent = !node.isRoot();
        int numNeighbours = this.collectNeighbourChanges(node, null);
        this.merge.reset(this.neighbourChanges, numNeighbours, reference);
        this.numSampledSites = 0;

        int mergedSite = this.merge.nextSite();
        long visitedSite = this.findNextVisitedSite(-1, this.numSites);
        while (mergedSite != Integer.MAX_VALUE || visitedSite < this.numSites) {
            int site;
            int state;

            if (mergedSite <= visitedSite) {
                site = mergedSite;
                int agreedState = this.findAgreedState(numNeighbours);
                if (agreedState < 0) {
                    state = this.drawDisagreeingState(hasParent, node.getChildCount());
                } else if (visitedSite == site) {
                    state = this.drawThinnedState(agreedState);
                } else {
                    state = agreedState;
                }

                if (visitedSite == site) {
                    visitedSite = this.findNextVisitedSite(visitedSite, this.numSites);
                }
                mergedSite = this.merge.nextSite();
            } else {
                // every neighbour is in its reference state
                site = (int) visitedSite;
                state = this.drawThinnedState(reference[site]);
                visitedSite = this.findNextVisitedSite(visitedSite, this.numSites);
            }

            if (state != reference[site]) {
                this.appendSampledState(site, state);
            }
        }

        return StateChanges.ofSorted(node.getNr(), this.sampledSites, this.sampledStates, this.numSampledSites);
    }

    /**
     * Collects the changes of the parent, unless the node is the root, and of the children of
     * the given node, followed by the given extra changes if they are not null. Returns the
     * number of neighbours, without the extra changes.
     */
    private int collectNeighbourChanges(Node node, StateChanges extraChanges) {
        int capacity = node.getChildCount() + 2;
        if (this.neighbourChanges.length < capacity) {
            this.neighbourChanges = new StateChanges[capacity];
        }

        int numNeighbours = 0;
        if (!node.isRoot()) {
            this.neighbourChanges[numNeighbours++] = this.nodeStates.getChanges(node.getParent());
        }
        for (Node child : node.getChildren()) {
            this.neighbourChanges[numNeighbours++] = this.nodeStates.getChanges(child);
        }
        if (extraChanges != null) {
            this.neighbourChanges[numNeighbours] = extraChanges;
        }
        return numNeighbours;
    }

    /** Returns the state in which the first given number of lists of the merge agree at its current site, or -1 if they disagree. */
    private int findAgreedState(int numNeighbours) {
        int agreedState = this.merge.getState(0);
        for (int i = 1; i < numNeighbours; i++) {
            if (this.merge.getState(i) != agreedState) {
                return -1;
            }
        }
        return agreedState;
    }

    /** Draws the state at the current site of the merge from the full conditional given the states of the neighbours. */
    private int drawDisagreeingState(boolean hasParent, int numChildren) {
        double total = this.computeSiteWeights(hasParent, numChildren);
        return Categorical.drawState(this.siteWeights, 0, this.numStates, total);
    }

    /** Fills the weights w(x) of the current site of the merge, given the states of the neighbours, and returns their total. */
    private double computeSiteWeights(boolean hasParent, int numChildren) {
        // all rows of the root are the frequencies
        int upperRow = hasParent ? this.merge.getState(0) : 0;
        int firstChild = hasParent ? 1 : 0;

        double total = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            double weight = this.upperProbabilities[upperRow * this.numStates + state];
            for (int c = 0; c < numChildren; c++) {
                weight *= this.childProbabilities[c][state * this.numStates + this.merge.getState(firstChild + c)];
            }
            this.siteWeights[state] = weight;
            total += weight;
        }
        return total;
    }

    /**
     * Draws the state at a visited site at which the neighbours agree in the given state. The
     * site was visited with probability r, so it leaves with probability ε_a / r.
     */
    private int drawThinnedState(int agreedState) {
        if (Randomizer.nextDouble() * this.maxLeaveProbability >= this.leaveProbabilities[agreedState]) {
            return agreedState;
        }
        return Categorical.drawState(this.leaveWeights, agreedState * this.numStates, this.numStates, this.leaveTotals[agreedState]);
    }

    /**
     * Returns the next site after the given one that the thinning visits, where each site is
     * visited independently with probability r, or the given end if there is none.
     */
    private long findNextVisitedSite(long site, int end) {
        if (this.maxLeaveProbability <= 0.0) {
            return end;
        }
        if (this.maxLeaveProbability >= 1.0) {
            return Math.min(end, site + 1);
        }

        double numSkippedSites = Math.floor(Math.log(1.0 - Randomizer.nextDouble()) / this.logSkipProbability);
        if (numSkippedSites >= end - site - 1) {
            return end;
        }
        return site + 1 + (long) numSkippedSites;
    }

    private void appendSampledState(int site, int state) {
        this.sampledSites[this.numSampledSites] = site;
        this.sampledStates[this.numSampledSites] = state;
        this.numSampledSites++;
    }

    /**
     * Samples the latent sites of a tip, whose only neighbour is its parent. The ambiguous
     * sites are drawn exactly, and the missing sites, at which the conditional only depends
     * on the state of the parent, are thinned over the indices of the missing sites. The
     * observed sites keep their current states.
     */
    private StateChanges sampleTip(Node tip) {
        int[] reference = this.nodeStates.getReferenceSequence();
        StateChanges parentChanges = this.nodeStates.getChanges(tip.getParent());
        StateChanges tipChanges = this.nodeStates.getChanges(tip);
        int[] missingSites = this.tipCompatibility.getMissingSites(tip);
        int[] ambiguousSites = this.tipCompatibility.getAmbiguousSites(tip);
        int[] ambiguousStates = this.tipCompatibility.getAmbiguousStates(tip);
        this.numSampledSites = 0;

        // keep the observed sites

        for (int i = 0; i < tipChanges.size(); i++) {
            if (!this.tipCompatibility.isLatent(tip, tipChanges.getSite(i))) {
                this.appendSampledState(tipChanges.getSite(i), tipChanges.getState(i));
            }
        }

        // draw the ambiguous sites exactly

        for (int i = 0; i < ambiguousSites.length; i++) {
            int site = ambiguousSites[i];
            int parentState = parentChanges.getStateOrDefault(site, reference[site]);

            double total = 0.0;
            for (int state = 0; state < this.numStates; state++) {
                boolean isCompatible = (ambiguousStates[i] & (1 << state)) != 0;
                this.siteWeights[state] = isCompatible ? this.upperProbabilities[parentState * this.numStates + state] : 0.0;
                total += this.siteWeights[state];
            }

            int state = Categorical.drawState(this.siteWeights, 0, this.numStates, total);
            if (state != reference[site]) {
                this.appendSampledState(site, state);
            }
        }

        // thin the missing sites, whose unvisited sites take the state of the parent, which only differs from the reference at the changes of the parent

        int parentIndex = this.findNextMissingChange(parentChanges, 0, missingSites);
        long visitedIndex = this.findNextVisitedSite(-1, missingSites.length);
        while (parentIndex < parentChanges.size() || visitedIndex < missingSites.length) {
            int parentSite = parentIndex < parentChanges.size() ? parentChanges.getSite(parentIndex) : Integer.MAX_VALUE;
            int visitedSite = visitedIndex < missingSites.length ? missingSites[(int) visitedIndex] : Integer.MAX_VALUE;
            int site = Math.min(parentSite, visitedSite);

            int state = reference[site];
            if (parentSite == site) {
                state = parentChanges.getState(parentIndex);
                parentIndex = this.findNextMissingChange(parentChanges, parentIndex + 1, missingSites);
            }
            if (visitedSite == site) {
                state = this.drawThinnedState(state);
                visitedIndex = this.findNextVisitedSite(visitedIndex, missingSites.length);
            }

            if (state != reference[site]) {
                this.appendSampledState(site, state);
            }
        }

        // the three parts are disjoint, but not sorted together
        return StateChanges.of(
                tip.getNr(),
                Arrays.copyOf(this.sampledSites, this.numSampledSites),
                Arrays.copyOf(this.sampledStates, this.numSampledSites)
        );
    }

    /** Returns the index of the first change at or after the given index whose site is missing, or the number of changes. */
    private int findNextMissingChange(StateChanges changes, int index, int[] missingSites) {
        while (index < changes.size() && Arrays.binarySearch(missingSites, changes.getSite(index)) < 0) {
            index++;
        }
        return index;
    }

    /* Density */

    /**
     * Returns the log probability with which the sampler draws the given states of the given
     * node, given as changes relative to the current reference sequence. This is the sum over
     * the sites of the log full conditional. For a tip, it is the density of its latent sites,
     * and negative infinity if the states are not compatible with its data.
     */
    public double computeLogDensity(Node node, StateChanges changes) {
        if (changes.getNodeNr() != node.getNr()) {
            throw new IllegalArgumentException("Changes are annotated with a different node.");
        }

        this.prepareNode(node);
        return node.isLeaf() ? this.computeTipLogDensity(node, changes) : this.computeInnerNodeLogDensity(node, changes);
    }

    /**
     * Merges the changes of the neighbours with the given changes. Every merged site
     * contributes its own log conditional, and every other site is in the reference state
     * like all neighbours, so these sites only contribute through their number per state.
     */
    private double computeInnerNodeLogDensity(Node node, StateChanges changes) {
        int[] reference = this.nodeStates.getReferenceSequence();
        boolean hasParent = !node.isRoot();
        int numNeighbours = this.collectNeighbourChanges(node, changes);
        this.merge.reset(this.neighbourChanges, numNeighbours + 1, reference);
        Arrays.fill(this.referenceStateCounts, 0);

        double logDensity = 0.0;
        for (int site = this.merge.nextSite(); site != Integer.MAX_VALUE; site = this.merge.nextSite()) {
            this.referenceStateCounts[reference[site]]++;

            int state = this.merge.getState(numNeighbours);
            int agreedState = this.findAgreedState(numNeighbours);
            if (agreedState < 0) {
                double total = this.computeSiteWeights(hasParent, node.getChildCount());
                logDensity += Math.log(this.siteWeights[state] / total);
            } else {
                logDensity += this.computeAgreedLogProbability(agreedState, state);
            }
        }

        int[] rootStateCounts = this.nodeStates.getRootStateCounts();
        for (int state = 0; state < this.numStates; state++) {
            int numUnmergedSites = rootStateCounts[state] - this.referenceStateCounts[state];
            if (numUnmergedSites != 0) {
                logDensity += numUnmergedSites * this.computeAgreedLogProbability(state, state);
            }
        }
        return logDensity;
    }

    /** Returns the log conditional of the given state at a site at which the neighbours agree in the given state. */
    private double computeAgreedLogProbability(int agreedState, int state) {
        double weight = state == agreedState ? this.stayWeights[agreedState] : this.leaveWeights[agreedState * this.numStates + state];
        return Math.log(weight / this.totalWeights[agreedState]);
    }

    /** Sums the log conditionals over the latent sites of the given tip, which costs O((M + A) log m). */
    private double computeTipLogDensity(Node tip, StateChanges changes) {
        int[] reference = this.nodeStates.getReferenceSequence();
        if (!this.tipCompatibility.isCompatible(tip, changes, reference)) {
            return Double.NEGATIVE_INFINITY;
        }

        StateChanges parentChanges = this.nodeStates.getChanges(tip.getParent());
        double logDensity = 0.0;

        for (int site : this.tipCompatibility.getMissingSites(tip)) {
            int parentState = parentChanges.getStateOrDefault(site, reference[site]);
            logDensity += this.computeAgreedLogProbability(parentState, changes.getStateOrDefault(site, reference[site]));
        }

        int[] ambiguousSites = this.tipCompatibility.getAmbiguousSites(tip);
        int[] ambiguousStates = this.tipCompatibility.getAmbiguousStates(tip);
        for (int i = 0; i < ambiguousSites.length; i++) {
            int site = ambiguousSites[i];
            int parentState = parentChanges.getStateOrDefault(site, reference[site]);

            double total = 0.0;
            for (int state = 0; state < this.numStates; state++) {
                if ((ambiguousStates[i] & (1 << state)) != 0) {
                    total += this.upperProbabilities[parentState * this.numStates + state];
                }
            }

            int state = changes.getStateOrDefault(site, reference[site]);
            logDensity += Math.log(this.upperProbabilities[parentState * this.numStates + state] / total);
        }

        return logDensity;
    }

}
