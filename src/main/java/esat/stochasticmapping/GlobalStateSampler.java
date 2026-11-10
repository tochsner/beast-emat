package esat.stochasticmapping;

import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import esat.helper.TipCompatibility;
import esat.state.NodeStates;
import esat.state.StateChanges;

import java.util.Arrays;

/**
 * Samples the states of every node at every site at once from their exact joint conditional
 * given the tree, the model and the tip data. The sites are independent, so each site is
 * sampled by a pruning pass up the tree and categorical draws down the tree. Sites with the
 * same pattern share their partials, which are computed for blocks of patterns at a time to
 * bound the memory.
 * <p>
 * The partials of a node are normalised to a maximum of one, as sampling only needs their
 * ratios. The sampler reads the model from the substitution process, which has to be updated
 * before.
 */
public class GlobalStateSampler {

    public static final int DEFAULT_BLOCK_SIZE = 256;

    /** A sample of the root sequence and of the changes of every node relative to it, where those of the root are empty. */
    public record Sample(int[] rootSequence, StateChanges[] changesAtNode) {
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

    // per node: the state at the current site, and the sampled sites and states at which it differs from the root
    private final int[] siteStates;
    private final int[][] changedSites;
    private final int[][] changedStates;
    private final int[] numChanges;

    private final double[] weights;

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
        this.changedStates = new int[this.numNodes][16];
        this.numChanges = new int[this.numNodes];

        this.weights = new double[this.numStates];
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
        this.collectNodesInPreorder();
        for (Node node : this.nodesInPreorder) {
            if (!node.isRoot()) {
                this.substitutionProcess.getTransitionProbabilities(node, this.transitionProbabilities[node.getNr()]);
            }
        }

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
        StateChanges[] changesAtNode = new StateChanges[this.numNodes];
        for (int nodeNr = 0; nodeNr < this.numNodes; nodeNr++) {
            changesAtNode[nodeNr] = StateChanges.of(
                    nodeNr,
                    Arrays.copyOf(this.changedSites[nodeNr], this.numChanges[nodeNr]),
                    Arrays.copyOf(this.changedStates[nodeNr], this.numChanges[nodeNr])
            );
        }
        return new Sample(rootSequence, changesAtNode);
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
            } else if (state != rootSequence[site]) {
                this.appendChange(nodeNr, site, state);
            }
        }
    }

    private void appendChange(int nodeNr, int site, int state) {
        if (this.numChanges[nodeNr] == this.changedSites[nodeNr].length) {
            this.changedSites[nodeNr] = Arrays.copyOf(this.changedSites[nodeNr], 2 * this.changedSites[nodeNr].length);
            this.changedStates[nodeNr] = Arrays.copyOf(this.changedStates[nodeNr], 2 * this.changedStates[nodeNr].length);
        }

        this.changedSites[nodeNr][this.numChanges[nodeNr]] = site;
        this.changedStates[nodeNr][this.numChanges[nodeNr]] = state;
        this.numChanges[nodeNr]++;
    }

}
