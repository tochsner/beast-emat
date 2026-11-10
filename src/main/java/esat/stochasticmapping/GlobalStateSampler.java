package esat.stochasticmapping;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.util.Randomizer;
import esat.helper.TipCompatibility;
import esat.state.NodeStates;
import esat.state.BranchChanges;

import java.util.Arrays;

/**
 * Samples the states of every node at every site at once, or at the sites of one pattern,
 * from their exact joint conditional given the tree, the model and the tip data. The sites
 * are independent, so each site is sampled by a pruning pass up the tree and categorical
 * draws down the tree. Sites with the same pattern share their partials, which are computed
 * for blocks of patterns at a time to bound the memory.
 * <p>
 * The partials of a node are normalised to a maximum of one, as sampling only needs their
 * ratios. The sampler reads the model from the substitution process, which has to be updated
 * before.
 */
public class GlobalStateSampler {

    public static final int DEFAULT_BLOCK_SIZE = 256;

    /** A sample of the root sequence and of the changes of every node relative to its parent, where those of the root are empty. */
    public record Sample(int[] rootSequence, BranchChanges[] changesAtNode) {
    }

    /**
     * A sample of the states at the sites of one pattern: the sites in ascending order, the
     * root state at each of them, and the changes of every node at these sites relative to
     * its parent, where those of the root are empty.
     */
    public record PatternSample(int[] sites, int[] rootStates, BranchChanges[] changesAtNode) {
    }

    private final TreeInterface tree;
    private final SubstitutionProcess substitutionProcess;
    private final TipCompatibility tipCompatibility;

    private final int numStates;
    private final int numSites;
    private final int numNodes;
    private final int numPatterns;
    private final int blockSize;

    // the sites of every pattern in ascending order
    private final int[][] sitesOfPattern;

    // the nodes in preorder, so that every parent comes before its children
    private final Node[] nodesInPreorder;

    // per node: the transition probabilities of the branch above it, and its partials of the patterns of the current block
    private final double[][] transitionProbabilities;
    private final double[][] partials;

    // per node: the state at the current site, and the sampled sites and states at which it differs from its parent, with the state of the parent
    private final int[] siteStates;
    private final int[][] changedSites;
    private final int[][] changedFromStates;
    private final int[][] changedToStates;
    private final int[] numChanges;

    // per node, for one pattern: the state at most of its sites, and the positions within the pattern and states of the other sites
    private final int[] defaultStates;
    private final int[][] exceptionPositions;
    private final int[][] exceptionStates;
    private final int[] numExceptions;

    // the sites and states of the changes of one node within a pattern, before they are copied
    private final int[] patternChangeSites;
    private final int[] patternChangeFromStates;
    private final int[] patternChangeToStates;

    private final double[] weights;
    private final double[] leaveWeights;

    public GlobalStateSampler(NodeStates nodeStates, SubstitutionProcess substitutionProcess) {
        this(nodeStates, substitutionProcess, DEFAULT_BLOCK_SIZE);
    }

    public GlobalStateSampler(NodeStates nodeStates, SubstitutionProcess substitutionProcess, int blockSize) {
        if (blockSize < 1) {
            throw new IllegalArgumentException("The block size must be positive.");
        }

        this.tree = nodeStates.getTree();
        this.substitutionProcess = substitutionProcess;
        this.tipCompatibility = substitutionProcess.getTipCompatibility();

        Alignment alignment = nodeStates.getAlignment();
        this.numStates = nodeStates.getNumStates();
        this.numSites = nodeStates.getNumSites();
        this.numNodes = this.tree.getNodeCount();
        this.numPatterns = alignment.getPatternCount();
        this.blockSize = Math.min(blockSize, this.numPatterns);
        if (substitutionProcess.getNumStates() != this.numStates) {
            throw new IllegalArgumentException("The node states have a different number of states than the substitution process.");
        }

        this.sitesOfPattern = computeSitesOfPattern(alignment);
        this.nodesInPreorder = new Node[this.numNodes];

        this.transitionProbabilities = new double[this.numNodes][this.numStates * this.numStates];
        this.partials = new double[this.numNodes][this.blockSize * this.numStates];

        this.siteStates = new int[this.numNodes];
        this.changedSites = new int[this.numNodes][16];
        this.changedFromStates = new int[this.numNodes][16];
        this.changedToStates = new int[this.numNodes][16];
        this.numChanges = new int[this.numNodes];

        this.defaultStates = new int[this.numNodes];
        this.exceptionPositions = new int[this.numNodes][16];
        this.exceptionStates = new int[this.numNodes][16];
        this.numExceptions = new int[this.numNodes];

        int maxPatternSize = 0;
        for (int[] sites : this.sitesOfPattern) {
            maxPatternSize = Math.max(maxPatternSize, sites.length);
        }
        this.patternChangeSites = new int[maxPatternSize];
        this.patternChangeFromStates = new int[maxPatternSize];
        this.patternChangeToStates = new int[maxPatternSize];

        this.weights = new double[this.numStates];
        this.leaveWeights = new double[this.numStates];
    }

    public int getPatternCount() {
        return this.numPatterns;
    }

    /** Returns the sites of every pattern of the given alignment in ascending order. */
    private static int[][] computeSitesOfPattern(Alignment alignment) {
        int[] numSitesOfPattern = new int[alignment.getPatternCount()];
        for (int site = 0; site < alignment.getSiteCount(); site++) {
            numSitesOfPattern[alignment.getPatternIndex(site)]++;
        }

        int[][] sitesOfPattern = new int[numSitesOfPattern.length][];
        for (int patternNr = 0; patternNr < numSitesOfPattern.length; patternNr++) {
            sitesOfPattern[patternNr] = new int[numSitesOfPattern[patternNr]];
        }

        int[] numFilledSites = new int[numSitesOfPattern.length];
        for (int site = 0; site < alignment.getSiteCount(); site++) {
            int patternNr = alignment.getPatternIndex(site);
            sitesOfPattern[patternNr][numFilledSites[patternNr]++] = site;
        }
        return sitesOfPattern;
    }

    /**
     * Samples the states of every node at every site from their joint conditional, which
     * takes O(P n K^2) for the partials of P patterns, O(L n K) for the draws at L sites, and
     * O(m log m) to sort the m changes of every node.
     */
    public Sample sample() {
        this.prepareTree();

        int[] rootSequence = new int[this.numSites];
        Arrays.fill(this.numChanges, 0);

        for (int firstPattern = 0; firstPattern < this.numPatterns; firstPattern += this.blockSize) {
            int numBlockPatterns = Math.min(this.blockSize, this.numPatterns - firstPattern);
            this.computePartials(firstPattern, numBlockPatterns);

            for (int i = 0; i < numBlockPatterns; i++) {
                for (int site : this.sitesOfPattern[firstPattern + i]) {
                    this.sampleSite(site, i, rootSequence);
                }
            }
        }

        // the sites were sampled pattern by pattern, so the changes are sorted here
        BranchChanges[] changesAtNode = new BranchChanges[this.numNodes];
        for (int nodeNr = 0; nodeNr < this.numNodes; nodeNr++) {
            changesAtNode[nodeNr] = BranchChanges.of(
                    nodeNr,
                    Arrays.copyOf(this.changedSites[nodeNr], this.numChanges[nodeNr]),
                    Arrays.copyOf(this.changedFromStates[nodeNr], this.numChanges[nodeNr]),
                    Arrays.copyOf(this.changedToStates[nodeNr], this.numChanges[nodeNr])
            );
        }
        return new Sample(rootSequence, changesAtNode);
    }

    /** Collects the nodes in preorder and computes the transition probabilities of every branch. */
    private void prepareTree() {
        this.collectNodesInPreorder();
        for (Node node : this.nodesInPreorder) {
            if (!node.isRoot()) {
                this.substitutionProcess.getTransitionProbabilities(node, this.transitionProbabilities[node.getNr()]);
            }
        }
    }

    /** Collects the nodes in preorder without recursion, which could overflow the stack on deep trees. */
    private void collectNodesInPreorder() {
        Node[] stack = new Node[this.numNodes];
        int stackSize = 0;
        int numCollected = 0;

        stack[stackSize++] = this.tree.getRoot();
        while (stackSize > 0) {
            Node node = stack[--stackSize];
            this.nodesInPreorder[numCollected++] = node;
            for (Node child : node.getChildren()) {
                stack[stackSize++] = child;
            }
        }
    }

    /**
     * Computes the partials of every node for the given patterns, from the tips up. The
     * partial of a tip is one for its compatible states and zero otherwise, and the partial of
     * an inner node is L(x) = Π_c Σ_y P_c(x, y) L_c(y), normalised to a maximum of one.
     */
    private void computePartials(int firstPattern, int numBlockPatterns) {
        // the reversed preorder visits the children before their parent
        for (int k = this.numNodes - 1; k >= 0; k--) {
            Node node = this.nodesInPreorder[k];
            double[] nodePartials = this.partials[node.getNr()];

            if (node.isLeaf()) {
                for (int i = 0; i < numBlockPatterns; i++) {
                    int compatibleStates = this.tipCompatibility.getCompatibleStatesOfPattern(node, firstPattern + i);
                    for (int state = 0; state < this.numStates; state++) {
                        nodePartials[i * this.numStates + state] = (compatibleStates & (1 << state)) != 0 ? 1.0 : 0.0;
                    }
                }
            } else {
                Arrays.fill(nodePartials, 0, numBlockPatterns * this.numStates, 1.0);
                for (Node child : node.getChildren()) {
                    this.multiplyChildPartials(nodePartials, child, numBlockPatterns);
                }
            }

            this.normalisePartials(node, nodePartials, firstPattern, numBlockPatterns);
        }
    }

    /** Multiplies the given partials of a node by Σ_y P_c(x, y) L_c(y) of the given child. */
    private void multiplyChildPartials(double[] nodePartials, Node child, int numBlockPatterns) {
        double[] childProbabilities = this.transitionProbabilities[child.getNr()];
        double[] childPartials = this.partials[child.getNr()];

        for (int i = 0; i < numBlockPatterns; i++) {
            int offset = i * this.numStates;
            for (int state = 0; state < this.numStates; state++) {
                double sum = 0.0;
                for (int childState = 0; childState < this.numStates; childState++) {
                    sum += childProbabilities[state * this.numStates + childState] * childPartials[offset + childState];
                }
                nodePartials[offset + state] *= sum;
            }
        }
    }

    private void normalisePartials(Node node, double[] nodePartials, int firstPattern, int numBlockPatterns) {
        for (int i = 0; i < numBlockPatterns; i++) {
            int offset = i * this.numStates;

            double max = 0.0;
            for (int state = 0; state < this.numStates; state++) {
                max = Math.max(max, nodePartials[offset + state]);
            }
            if (!(max > 0.0) || !Double.isFinite(max)) {
                throw new IllegalStateException("The data of pattern " + (firstPattern + i) + " below node "
                        + node.getNr() + " are impossible under the tree and the model.");
            }

            for (int state = 0; state < this.numStates; state++) {
                nodePartials[offset + state] /= max;
            }
        }
    }

    /**
     * Draws the states of every node at the given site from the root down, using the partials
     * of the pattern with the given index in the current block. The root is drawn
     * proportional to π(x) L(x) and every other node proportional to P(a, x) L(x), where a is
     * the state of its parent.
     */
    private void sampleSite(int site, int blockIndex, int[] rootSequence) {
        int offset = blockIndex * this.numStates;
        double[] frequencies = this.substitutionProcess.getFrequencies();

        for (Node node : this.nodesInPreorder) {
            int nodeNr = node.getNr();
            double[] nodePartials = this.partials[nodeNr];

            double total = 0.0;
            for (int state = 0; state < this.numStates; state++) {
                double prior = node.isRoot()
                        ? frequencies[state]
                        : this.transitionProbabilities[nodeNr][this.siteStates[node.getParent().getNr()] * this.numStates + state];
                this.weights[state] = prior * nodePartials[offset + state];
                total += this.weights[state];
            }

            int state = Categorical.drawState(this.weights, 0, this.numStates, total);
            this.siteStates[nodeNr] = state;

            if (node.isRoot()) {
                rootSequence[site] = state;
            } else if (state != this.siteStates[node.getParent().getNr()]) {
                this.appendChange(nodeNr, site, this.siteStates[node.getParent().getNr()], state);
            }
        }
    }

    private void appendChange(int nodeNr, int site, int fromState, int toState) {
        if (this.numChanges[nodeNr] == this.changedSites[nodeNr].length) {
            this.changedSites[nodeNr] = Arrays.copyOf(this.changedSites[nodeNr], 2 * this.changedSites[nodeNr].length);
            this.changedFromStates[nodeNr] = Arrays.copyOf(this.changedFromStates[nodeNr], 2 * this.changedFromStates[nodeNr].length);
            this.changedToStates[nodeNr] = Arrays.copyOf(this.changedToStates[nodeNr], 2 * this.changedToStates[nodeNr].length);
        }

        this.changedSites[nodeNr][this.numChanges[nodeNr]] = site;
        this.changedFromStates[nodeNr][this.numChanges[nodeNr]] = fromState;
        this.changedToStates[nodeNr][this.numChanges[nodeNr]] = toState;
        this.numChanges[nodeNr]++;
    }

    /* Single Pattern */

    /**
     * Samples the states of every node at the sites of the given pattern from their joint
     * conditional, given the states at all other sites. The sites of a pattern share their
     * partials, so the states of a node at the sites where its parent has the same state are
     * independent and identically distributed. Every node therefore has a default state, the
     * most probable one given the default state of its parent, and lists the sites where it
     * deviates as exceptions. The sites where the node leaves the default state of its parent
     * are found by geometric skipping, so a pattern takes O(n K^2) for the partials and
     * O(n K + e) for the e exceptions, independently of its number of sites.
     */
    public PatternSample samplePattern(int patternNr) {
        this.prepareTree();
        this.computePartials(patternNr, 1);

        int[] sites = this.sitesOfPattern[patternNr];
        for (Node node : this.nodesInPreorder) {
            this.samplePatternStates(node, sites.length);
        }

        int rootNr = this.tree.getRoot().getNr();
        int[] rootStates = new int[sites.length];
        Arrays.fill(rootStates, this.defaultStates[rootNr]);
        for (int k = 0; k < this.numExceptions[rootNr]; k++) {
            rootStates[this.exceptionPositions[rootNr][k]] = this.exceptionStates[rootNr][k];
        }

        BranchChanges[] changesAtNode = new BranchChanges[this.numNodes];
        for (Node node : this.nodesInPreorder) {
            changesAtNode[node.getNr()] = node.isRoot()
                    ? BranchChanges.empty(node.getNr())
                    : this.collectChangesToParent(node.getNr(), node.getParent().getNr(), sites);
        }
        return new PatternSample(sites.clone(), rootStates, changesAtNode);
    }

    /**
     * Samples the default state and the exceptions of the given node at the given number of
     * sites of the current pattern, given those of its parent. The positions where the parent
     * deviates from its default state are drawn exactly given the state of the parent there,
     * and at all other positions the node leaves its default state with the same probability,
     * so these are visited by geometric skipping.
     */
    private void samplePatternStates(Node node, int numPatternSites) {
        int nodeNr = node.getNr();
        int parentNr = node.isRoot() ? -1 : node.getParent().getNr();
        this.numExceptions[nodeNr] = 0;

        // the distribution of the state at the positions where the parent has its default state

        double total = this.computePatternWeights(node, node.isRoot() ? -1 : this.defaultStates[parentNr], this.weights);
        int defaultState = 0;
        for (int state = 1; state < this.numStates; state++) {
            if (this.weights[state] > this.weights[defaultState]) {
                defaultState = state;
            }
        }
        this.defaultStates[nodeNr] = defaultState;

        double leaveTotal = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            this.leaveWeights[state] = state == defaultState ? 0.0 : this.weights[state];
            leaveTotal += this.leaveWeights[state];
        }
        double logStayProbability = Math.log1p(-leaveTotal / total);

        // merge the exceptions of the parent with the positions where the node leaves its default state

        int numParentExceptions = node.isRoot() ? 0 : this.numExceptions[parentNr];
        int k = 0;
        int nextLeave = this.findNextLeave(-1, logStayProbability, numPatternSites);
        while (k < numParentExceptions || nextLeave < numPatternSites) {
            int parentPosition = k < numParentExceptions ? this.exceptionPositions[parentNr][k] : Integer.MAX_VALUE;

            if (parentPosition <= nextLeave) {
                double siteTotal = this.computePatternWeights(node, this.exceptionStates[parentNr][k], this.weights);
                int state = Categorical.drawState(this.weights, 0, this.numStates, siteTotal);
                if (state != defaultState) {
                    this.appendException(nodeNr, parentPosition, state);
                }
                k++;

                // a leave at a position where the parent deviates is replaced by the exact draw
                if (parentPosition == nextLeave) {
                    nextLeave = this.findNextLeave(nextLeave, logStayProbability, numPatternSites);
                }
            } else {
                int state = Categorical.drawState(this.leaveWeights, 0, this.numStates, leaveTotal);
                this.appendException(nodeNr, nextLeave, state);
                nextLeave = this.findNextLeave(nextLeave, logStayProbability, numPatternSites);
            }
        }
    }

    /**
     * Writes the weights π(x) L(x) of the root or P(a, x) L(x) of any other node at the
     * current pattern into the given array, where a is the given state of the parent, and
     * returns their sum.
     */
    private double computePatternWeights(Node node, int parentState, double[] stateWeights) {
        int nodeNr = node.getNr();
        double[] nodePartials = this.partials[nodeNr];
        double[] frequencies = this.substitutionProcess.getFrequencies();

        double total = 0.0;
        for (int state = 0; state < this.numStates; state++) {
            double prior = node.isRoot()
                    ? frequencies[state]
                    : this.transitionProbabilities[nodeNr][parentState * this.numStates + state];
            stateWeights[state] = prior * nodePartials[state];
            total += stateWeights[state];
        }
        return total;
    }

    /**
     * Returns the next position after the given one at which a node leaves its default state,
     * where it stays with the given log probability at every position, or the given number of
     * positions if there is none.
     */
    private int findNextLeave(int position, double logStayProbability, int numPositions) {
        if (logStayProbability == 0.0) {
            return numPositions;
        }

        double skip = Math.floor(Math.log(1.0 - Randomizer.nextDouble()) / logStayProbability);
        return (int) Math.min(position + 1.0 + skip, numPositions);
    }

    private void appendException(int nodeNr, int position, int state) {
        if (this.numExceptions[nodeNr] == this.exceptionPositions[nodeNr].length) {
            this.exceptionPositions[nodeNr] = Arrays.copyOf(this.exceptionPositions[nodeNr], 2 * this.exceptionPositions[nodeNr].length);
            this.exceptionStates[nodeNr] = Arrays.copyOf(this.exceptionStates[nodeNr], 2 * this.exceptionStates[nodeNr].length);
        }

        this.exceptionPositions[nodeNr][this.numExceptions[nodeNr]] = position;
        this.exceptionStates[nodeNr][this.numExceptions[nodeNr]] = state;
        this.numExceptions[nodeNr]++;
    }

    /**
     * Returns the changes of the given node at the sites of the current pattern relative to its
     * parent. If the node shares the default state of its parent, it can only differ at the
     * exceptions of either, otherwise every position is compared.
     */
    private BranchChanges collectChangesToParent(int nodeNr, int parentNr, int[] sites) {
        int numNodeExceptions = this.numExceptions[nodeNr];
        int numParentExceptions = this.numExceptions[parentNr];
        int nodeDefaultState = this.defaultStates[nodeNr];
        int parentDefaultState = this.defaultStates[parentNr];
        boolean haveSameDefault = nodeDefaultState == parentDefaultState;
        int size = 0;

        int i = 0;
        int j = 0;
        int position = 0;
        while (haveSameDefault ? (i < numNodeExceptions || j < numParentExceptions) : position < sites.length) {
            int nodePosition = i < numNodeExceptions ? this.exceptionPositions[nodeNr][i] : Integer.MAX_VALUE;
            int parentPosition = j < numParentExceptions ? this.exceptionPositions[parentNr][j] : Integer.MAX_VALUE;
            if (haveSameDefault) {
                position = Math.min(nodePosition, parentPosition);
            }

            int nodeState = nodePosition == position ? this.exceptionStates[nodeNr][i++] : nodeDefaultState;
            int parentState = parentPosition == position ? this.exceptionStates[parentNr][j++] : parentDefaultState;
            if (nodeState != parentState) {
                this.patternChangeSites[size] = sites[position];
                this.patternChangeFromStates[size] = parentState;
                this.patternChangeToStates[size] = nodeState;
                size++;
            }

            if (!haveSameDefault) {
                position++;
            }
        }

        return BranchChanges.ofSorted(nodeNr, this.patternChangeSites, this.patternChangeFromStates, this.patternChangeToStates, size);
    }

}
