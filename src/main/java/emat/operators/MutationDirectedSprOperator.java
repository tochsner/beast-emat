package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.helper.SiteChanges;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Description("Regrafts a random subtree preferably where the sequence is close to the sequence of the subtree, " +
        "using the mutations of the EMAT as a parsimony map (mutation-directed SPR, docs/mcmc-moves.md §8.3). " +
        "Most proposals only explore the neighbourhood of the current attachment point, whose extent is drawn " +
        "from a geometric distribution. The root is never changed.")
public class MutationDirectedSprOperator extends SubtreePruneRegraftOperator {

    public enum SubtreeSelection {
        UNIFORM, BRANCH_LENGTH, MOVED_NUM_NODES, MOVED_PATH_LENGTH, MOVED_SQUARED_PATH_LENGTH
    }

    final public Input<Double> annealingInput = new Input<>("annealing", "the power f to which the grafting density is raised, which makes the proposals less eager", 0.8);
    final public Input<Double> fullExplorationProbabilityInput = new Input<>("fullExplorationProbability", "the probability of exploring the whole tree instead of the neighbourhood of the current attachment point", 0.01);
    final public Input<Double> windowProbabilityInput = new Input<>("windowProbability", "the parameter q of the geometric number of additional differences, q^j (1 - q), that the neighbourhood may have compared to the current attachment point. With 0, the neighbourhood never has more differences than the current attachment point", 0.0);

    final public Input<SubtreeSelection> subtreeSelectionInput = new Input<>("subtreeSelection", "the strategy for choosing the subtree to regraft: UNIFORM picks every subtree with the same probability, " +
            "BRANCH_LENGTH proportional to the length of the branch above it, and MOVED_NUM_NODES, MOVED_PATH_LENGTH and MOVED_SQUARED_PATH_LENGTH " +
            "proportional to the running mean distance of its accepted moves, measured as the number of nodes, the length or the squared length " +
            "of the path between the old and the new attachment point", SubtreeSelection.UNIFORM, SubtreeSelection.values());
    final public Input<Double> moveDistancePseudocountInput = new Input<>("moveDistancePseudocount", "the number of pseudo-moves with the mean distance of all recorded moves " +
            "that the running mean distance of every subtree starts with", 10.0);
    final public Input<Boolean> countRejectedMovesInput = new Input<>("countRejectedMoves", "whether the running mean distance of a subtree also counts its rejected proposals " +
            "as moves of distance zero, which makes it the expected distance per proposal. Otherwise, it is the mean distance of the accepted moves", false);
    final public Input<Boolean> scaleWeightByLogNumTipsInput = new Input<>("scaleWeightByLogNumTips", "whether to scale the weight by the natural logarithm of the number of tips n, so that the operator is selected with weight * ln(n). Otherwise, the weight is used as given", false);

    private static final int INITIAL_CAPACITY = 64;

    // the factor ln(n) or one by which the weight of the operator is scaled
    double weightFactor;

    SubtreeSelection subtreeSelection;
    double annealing;
    double fullExplorationProbability;
    double windowProbability;
    double moveDistancePseudocount;
    boolean countRejectedMoves;

    // the selection weights of all nodes in the current state, which are zero for nodes that cannot be picked
    double[] selectionWeights;

    // the state of the current proposal: X, the joined mutations of the branch above S, and the constant part log(λ(X) / 3L) of the log grafting density
    Node x;
    Node sibling;
    List<Mutation> joinedMutations;
    double logRatePerSite;

    // the sequence of X as its differences to the reference sequence, which hold the state of X as end state
    SiteChanges subtreeSequenceChanges;
    int[] referenceSequence;

    // the explored regions of the current proposal, with their log weights and then their weights relative to the largest one
    int numRegions;
    Node[] regionNodes = new Node[INITIAL_CAPACITY];
    int[] regionIndices = new int[INITIAL_CAPACITY];
    int[] regionNumDifferences = new int[INITIAL_CAPACITY];
    double[] regionLowerHeights = new double[INITIAL_CAPACITY];
    double[] regionUpperHeights = new double[INITIAL_CAPACITY];
    double[] regionWeights = new double[INITIAL_CAPACITY];

    // the junctions still to explore, i.e. nodes of the pruned tree where branches meet, with their number of differences to X
    int numJunctions;
    Node[] junctionNodes = new Node[INITIAL_CAPACITY];
    int[] junctionNumDifferences = new int[INITIAL_CAPACITY];

    // per node, the exploration that walked the run of regions at the top or the bottom of its branch
    int explorationStamp;
    int[] topRunStamps;
    int[] bottomRunStamps;
    int threshold;

    // the subtree root of the current proposal, its old attachment point, its new grafting point and the distance between them, which are recorded as a move once accepted
    Node pickedSubtreeRoot;
    double oldParentHeight;
    GraftingPoint proposedGraftingPoint;
    int proposedNumNodes;
    double proposedPathLength;

    // per node, the number and the total distance of its recorded moves, and the same over all nodes
    int[] numMoves;
    double[] totalMoveDistances;
    long numAllMoves;
    double totalAllMoveDistance;

    // per node, the number of steps from the old sibling to the node, or -1 if it is no ancestor of the old sibling in the pruned tree
    int[] ancestorSteps;

    // the diagnostics of the proposals so far: how often each node was picked, and the distances of the proposed and the accepted moves
    int[] numPicks;
    long numMeasuredProposals;
    long numSameBranchProposals;
    long totalProposedNumNodes;
    double totalProposedPathLength;
    long numMeasuredAccepts;
    long totalAcceptedNumNodes;
    double totalAcceptedPathLength;

    // the number of local explorations so far and the number of regions they explored
    long numLocalExplorations;
    long numLocalRegions;

    /**
     * A part of a branch of the pruned tree with a constant sequence: the node below the
     * branch, the index of the part counted from the top of the branch, the height range
     * of the part that is older than X, and the number of sites where its sequence differs
     * from the sequence of X, leaving out the sites where the data of X is missing.
     */
    record Region(Node node, int index, double lowerHeight, double upperHeight, int numDifferences) {
    }

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        // the grafting density of the reverse move is computed on the pruned tree of the current state, which must keep its mutations
        if (this.resampleNeighbourhood) {
            throw new IllegalArgumentException("mdSPR does not support resampleNeighbourhood, as it changes the mutations of the pruned tree.");
        }

        this.subtreeSelection = this.subtreeSelectionInput.get();
        this.annealing = this.annealingInput.get();
        this.fullExplorationProbability = this.fullExplorationProbabilityInput.get();
        this.windowProbability = this.windowProbabilityInput.get();
        if (this.windowProbability < 0.0 || this.windowProbability >= 1.0) {
            throw new IllegalArgumentException("windowProbability must be at least 0 and less than 1.");
        }
        this.moveDistancePseudocount = this.moveDistancePseudocountInput.get();
        if (this.moveDistancePseudocount <= 0.0) {
            throw new IllegalArgumentException("moveDistancePseudocount must be positive.");
        }
        this.countRejectedMoves = this.countRejectedMovesInput.get();

        this.subtreeSequenceChanges = new SiteChanges(this.mutations.getReferenceSequence().length);

        int numNodes = this.tree.getNodeCount();
        this.selectionWeights = new double[numNodes];
        this.topRunStamps = new int[numNodes];
        this.bottomRunStamps = new int[numNodes];
        this.numMoves = new int[numNodes];
        this.totalMoveDistances = new double[numNodes];
        this.ancestorSteps = new int[numNodes];
        this.numPicks = new int[numNodes];
        Arrays.fill(this.ancestorSteps, -1);

        this.weightFactor = this.scaleWeightByLogNumTipsInput.get() ? Math.log(this.tree.getLeafNodeCount()) : 1.0;
    }

    /**
     * Returns the weight, scaled by the logarithm of the number of tips if requested, which
     * lets the share of the proposals grow with the size of the tree.
     */
    @Override
    public double getWeight() {
        return super.getWeight() * this.weightFactor;
    }

    /**
     * Picks X with probability proportional to its selection weight among all nodes where
     * neither X nor its parent is the root.
     */
    @Override
    protected Node pickSubtreeRoot() {
        this.pickedSubtreeRoot = null;
        this.proposedGraftingPoint = null;
        double threshold = Randomizer.nextDouble() * this.computeSelectionWeights();

        Node lastCandidate = null;
        double cumulativeWeight = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            double weight = this.selectionWeights[node.getNr()];
            if (weight <= 0.0) {
                continue;
            }

            lastCandidate = node;
            cumulativeWeight += weight;
            if (threshold < cumulativeWeight) {
                return this.markPicked(node);
            }
        }

        // only reached through rounding, or without any candidate
        return this.markPicked(lastCandidate);
    }

    private Node markPicked(Node node) {
        this.pickedSubtreeRoot = node;
        if (node != null) {
            this.numPicks[node.getNr()]++;
        }
        return node;
    }

    /**
     * Returns the log of the selection weight of X over the total weight of the candidates.
     * With UNIFORM, the number of candidates stays the same under moves that keep the root,
     * so the choice is symmetric. With BRANCH_LENGTH, the move changes the lengths of the
     * branches around the old and the new attachment point, and the nodes below the root.
     * The mean move distances only change between proposals, but the move can still change
     * which nodes hang below the root and hence the total weight.
     */
    @Override
    protected double computeLogSubtreeRootProbability(Node x) {
        if (this.subtreeSelection == SubtreeSelection.UNIFORM) {
            return 0.0;
        }

        double totalWeight = this.computeSelectionWeights();
        return Math.log(this.selectionWeights[x.getNr()] / totalWeight);
    }

    /**
     * Fills the selection weights of all nodes for the current state and returns their
     * total. The root and its children get the weight zero.
     */
    private double computeSelectionWeights() {
        double totalWeight = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            double weight = node.isRoot() || node.getParent().isRoot() ? 0.0 : this.computeSelectionWeight(node);
            this.selectionWeights[node.getNr()] = weight;
            totalWeight += weight;
        }
        return totalWeight;
    }

    /** Computes the unnormalised probability of picking the given node as X. */
    private double computeSelectionWeight(Node node) {
        return switch (this.subtreeSelection) {
            case UNIFORM -> 1.0;
            case BRANCH_LENGTH -> node.getLength();
            case MOVED_NUM_NODES, MOVED_PATH_LENGTH, MOVED_SQUARED_PATH_LENGTH -> this.computeMeanMoveDistance(node);
        };
    }

    /**
     * Computes the running mean distance of the recorded moves of the given node, which
     * starts with a pseudocount of moves at the mean distance of all recorded moves. Without
     * any distance yet, every node gets the same weight.
     */
    private double computeMeanMoveDistance(Node node) {
        if (this.totalAllMoveDistance <= 0.0) {
            return 1.0;
        }

        double priorMeanDistance = this.totalAllMoveDistance / this.numAllMoves;
        int nr = node.getNr();
        return (this.moveDistancePseudocount * priorMeanDistance + this.totalMoveDistances[nr])
                / (this.moveDistancePseudocount + this.numMoves[nr]);
    }

    /**
     * Cuts the branches of the pruned tree at their mutations into regions of constant
     * sequence, picks a region with probability proportional to its length times the
     * grafting density g at its midpoint, and draws the height uniformly within it. Under
     * Jukes–Cantor with rate λ spread over L sites, the density of attaching at distance
     * τ above X to a sequence with N differences at the observed sites of X is
     * g = [exp(-λτ) (λτ / 3L)^N]^f (Eq. 44), where λ is the rate of the Jukes-Cantor mapping.
     * <p>
     * With a small fixed probability, all regions of the pruned tree are candidates. Their
     * total weight is the same before and after the move, so it cancels. Otherwise, only the
     * regions connected to the current attachment point through regions with at most T
     * differences are candidates. The threshold is T = max(1, N) + j, where N is the number
     * of differences at the current attachment point and j is geometric with probability
     * q^j (1 - q). The reverse move needs the same threshold to find the same regions, whose
     * total weight then cancels as well. From the new attachment point with N' differences,
     * it draws that threshold with j' = T - max(1, N'), so the Hastings ratio gains the
     * factor q^(max(1, N) - max(1, N')). With q = 0, a move to fewer differences cannot be
     * reversed and is rejected.
     */
    @Override
    protected GraftingPoint proposeGraftingPoint(Node x) {
        this.x = x;
        this.referenceSequence = this.mutations.getReferenceSequence();
        MutationPaths.collectSequence(this.mutations, x, this.subtreeSequenceChanges);
        this.logRatePerSite = Math.log(this.subtreeTotalRate / (3.0 * this.referenceSequence.length));

        // in the pruned tree, the branches above P and above S are joined into one branch above S

        Node parent = x.getParent();
        this.sibling = this.getOtherChild(parent, x);
        this.joinedMutations = new ArrayList<>(this.mutations.getMutations(parent));
        this.joinedMutations.addAll(this.mutations.getMutations(this.sibling));

        // the current attachment point lies on the joined branch, where it differs from X by the changes on the P–X branch

        int oldIndex = 0;
        while (oldIndex < this.joinedMutations.size() && this.joinedMutations.get(oldIndex).time() >= parent.getHeight()) {
            oldIndex++;
        }

        MutationPaths.collectChanges(this.mutations, parent, x, x.getHeight(), this.subtreeChanges);
        int oldNumDifferences = this.countObservedDifferences(this.subtreeChanges);

        Region oldRegion = new Region(
                this.sibling, oldIndex,
                this.getLowerHeight(this.sibling, oldIndex), this.getUpperHeight(this.sibling, oldIndex),
                oldNumDifferences
        );

        // explore the candidate regions and pick one

        boolean isFullExploration = Randomizer.nextDouble() < this.fullExplorationProbability;
        int oldMinThreshold = Math.max(1, oldNumDifferences);
        int threshold = isFullExploration ? Integer.MAX_VALUE : this.sampleThreshold(oldMinThreshold);

        this.exploreRegions(oldRegion, threshold);
        if (!isFullExploration) {
            this.numLocalExplorations++;
            this.numLocalRegions += this.numRegions;
        }
        double logTotalWeight = this.normaliseRegionWeights();

        int newRegionNr = this.sampleRegion();
        Region newRegion = this.getRegion(newRegionNr);
        double newParentHeight = newRegion.lowerHeight() + Randomizer.nextDouble() * (newRegion.upperHeight() - newRegion.lowerHeight());

        // the reverse move has to draw the same threshold from the new attachment point

        double logThresholdRatio = 0.0;
        int newMinThreshold = Math.max(1, newRegion.numDifferences());
        if (!isFullExploration && newMinThreshold != oldMinThreshold) {
            if (this.windowProbability == 0.0) {
                // the reverse move cannot propose the current attachment point
                return null;
            }
            logThresholdRatio = (oldMinThreshold - newMinThreshold) * Math.log(this.windowProbability);
        }

        // the density of a point in a region is g(midpoint) divided by the total weight of the candidates

        double logForwardDensity = this.computeLogDensity(newRegion) - logTotalWeight;
        double logBackwardDensity = this.computeLogDensity(oldRegion) - logTotalWeight;

        this.oldParentHeight = parent.getHeight();
        this.proposedGraftingPoint = new GraftingPoint(newRegion.node(), newParentHeight, logBackwardDensity - logForwardDensity + logThresholdRatio);
        this.measureProposedMove();
        return this.proposedGraftingPoint;
    }

    /**
     * Draws the threshold of a local exploration: the given minimum plus a geometric number
     * of additional differences j with probability q^j (1 - q).
     */
    private int sampleThreshold(int minThreshold) {
        if (this.windowProbability == 0.0) {
            return minThreshold;
        }

        double numAdditionalDifferences = Math.floor(Math.log(Randomizer.nextDouble()) / Math.log(this.windowProbability));
        return minThreshold + (int) Math.min(numAdditionalDifferences, Integer.MAX_VALUE - minThreshold);
    }

    /** Returns the mean number of candidate regions of the local explorations so far. */
    public double getMeanNumLocalRegions() {
        return this.numLocalExplorations == 0 ? Double.NaN : (double) this.numLocalRegions / this.numLocalExplorations;
    }

    /* Regions */

    /**
     * Collects the regions older than X that are connected to the given region through
     * regions with at most the given number of differences. Crossing a mutation changes
     * the number of differences at its site, and crossing a node keeps it.
     * <p>
     * The number of differences of a region does not depend on the path to it, so the
     * regions within the threshold on a branch form runs. Walking down from the top of a
     * branch finds its first run, and walking up from the bottom its last run, which are
     * either the same run spanning the whole branch or disjoint. Hence, each branch is walked
     * at most once from each end, and only the junctions at the ends of the branches need a
     * stack. The regions are written into the region buffers.
     */
    private void exploreRegions(Region start, int threshold) {
        this.startExploration(threshold);

        this.addRegion(start.node(), start.index(), start.numDifferences());
        this.walkUp(start.node(), start.index(), start.numDifferences());
        this.walkDown(start.node(), start.index(), start.numDifferences());

        while (this.numJunctions > 0) {
            this.numJunctions--;
            Node junction = this.junctionNodes[this.numJunctions];
            int numDifferences = this.junctionNumDifferences[this.numJunctions];

            List<Node> children = junction.getChildrenMutable();
            for (int i = 0; i < children.size(); i++) {
                this.enterFromTop(this.getPrunedChild(children.get(i)), numDifferences);
            }

            if (!junction.isRoot()) {
                this.enterFromBottom(junction, numDifferences);
            }
        }
    }

    /** Resets the region buffers and the junction stack, and starts a new stamp for the walked runs. */
    private void startExploration(int threshold) {
        this.threshold = threshold;
        this.numRegions = 0;
        this.numJunctions = 0;

        // the stamps only need resetting once the counter runs out
        if (this.explorationStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.topRunStamps, 0);
            Arrays.fill(this.bottomRunStamps, 0);
            this.explorationStamp = 0;
        }
        this.explorationStamp++;
    }

    /** Walks the run of regions at the top of the branch above the given node, unless it was already walked. */
    private void enterFromTop(Node node, int numDifferences) {
        if (this.topRunStamps[node.getNr()] == this.explorationStamp) {
            return;
        }
        this.topRunStamps[node.getNr()] = this.explorationStamp;

        if (numDifferences <= this.threshold && this.addRegion(node, 0, numDifferences)) {
            this.walkDown(node, 0, numDifferences);
        }
    }

    /** Walks the run of regions at the bottom of the branch above the given node, unless it was already walked. */
    private void enterFromBottom(Node node, int numDifferences) {
        if (this.bottomRunStamps[node.getNr()] == this.explorationStamp) {
            return;
        }
        this.bottomRunStamps[node.getNr()] = this.explorationStamp;

        int index = this.getBranchMutations(node).size();
        if (numDifferences <= this.threshold && this.addRegion(node, index, numDifferences)) {
            this.walkUp(node, index, numDifferences);
        }
    }

    /**
     * Adds the regions below the given region on the branch above the given node until one
     * exceeds the threshold or lies below X. If the walk reaches the node, the junction
     * there is explored next.
     */
    private void walkDown(Node node, int index, int numDifferences) {
        List<Mutation> branchMutations = this.getBranchMutations(node);

        for (int i = index; i < branchMutations.size(); i++) {
            numDifferences += this.computeDifferenceChange(branchMutations.get(i));
            if (numDifferences > this.threshold || !this.addRegion(node, i + 1, numDifferences)) {
                return;
            }
        }

        this.bottomRunStamps[node.getNr()] = this.explorationStamp;

        // junctions not older than X only lead to regions below X
        if (node.getHeight() > this.x.getHeight()) {
            this.pushJunction(node, numDifferences);
        }
    }

    /**
     * Adds the regions above the given region on the branch above the given node until one
     * exceeds the threshold. If the walk reaches the top of the branch, the junction there
     * is explored next.
     */
    private void walkUp(Node node, int index, int numDifferences) {
        List<Mutation> branchMutations = this.getBranchMutations(node);

        for (int i = index; i > 0; i--) {
            numDifferences -= this.computeDifferenceChange(branchMutations.get(i - 1));
            if (numDifferences > this.threshold || !this.addRegion(node, i - 1, numDifferences)) {
                return;
            }
        }

        this.topRunStamps[node.getNr()] = this.explorationStamp;
        this.pushJunction(this.getPrunedParent(this.x, node), numDifferences);
    }

    /**
     * Adds the region with the given index on the branch above the given node to the region
     * buffers, together with its log weight. Returns false without adding it if no part of
     * it is older than X.
     */
    private boolean addRegion(Node node, int index, int numDifferences) {
        double lowerHeight = this.getLowerHeight(node, index);
        double upperHeight = this.getUpperHeight(node, index);
        if (upperHeight <= lowerHeight) {
            return false;
        }

        if (this.numRegions == this.regionNodes.length) {
            int capacity = 2 * this.numRegions;
            this.regionNodes = Arrays.copyOf(this.regionNodes, capacity);
            this.regionIndices = Arrays.copyOf(this.regionIndices, capacity);
            this.regionNumDifferences = Arrays.copyOf(this.regionNumDifferences, capacity);
            this.regionLowerHeights = Arrays.copyOf(this.regionLowerHeights, capacity);
            this.regionUpperHeights = Arrays.copyOf(this.regionUpperHeights, capacity);
            this.regionWeights = Arrays.copyOf(this.regionWeights, capacity);
        }

        int regionNr = this.numRegions++;
        this.regionNodes[regionNr] = node;
        this.regionIndices[regionNr] = index;
        this.regionNumDifferences[regionNr] = numDifferences;
        this.regionLowerHeights[regionNr] = lowerHeight;
        this.regionUpperHeights[regionNr] = upperHeight;
        this.regionWeights[regionNr] = this.computeLogWeight(lowerHeight, upperHeight, numDifferences);
        return true;
    }

    private void pushJunction(Node node, int numDifferences) {
        if (this.numJunctions == this.junctionNodes.length) {
            int capacity = 2 * this.numJunctions;
            this.junctionNodes = Arrays.copyOf(this.junctionNodes, capacity);
            this.junctionNumDifferences = Arrays.copyOf(this.junctionNumDifferences, capacity);
        }

        this.junctionNodes[this.numJunctions] = node;
        this.junctionNumDifferences[this.numJunctions] = numDifferences;
        this.numJunctions++;
    }

    private Region getRegion(int regionNr) {
        return new Region(
                this.regionNodes[regionNr], this.regionIndices[regionNr],
                this.regionLowerHeights[regionNr], this.regionUpperHeights[regionNr],
                this.regionNumDifferences[regionNr]
        );
    }

    /**
     * Returns the lower end of the region with the given index on the branch above the given
     * node: the mutation below it, or the node, truncated at the height of X.
     */
    private double getLowerHeight(Node node, int index) {
        List<Mutation> branchMutations = this.getBranchMutations(node);
        double lowerHeight = index == branchMutations.size() ? node.getHeight() : branchMutations.get(index).time();
        return Math.max(lowerHeight, this.x.getHeight());
    }

    /**
     * Returns the upper end of the region with the given index on the branch above the given
     * node: the mutation above it, or the top of the branch.
     */
    private double getUpperHeight(Node node, int index) {
        return index == 0
                ? this.getPrunedParent(this.x, node).getHeight()
                : this.getBranchMutations(node).get(index - 1).time();
    }

    /** Returns the mutations on the branch above the given node in the pruned tree. */
    private List<Mutation> getBranchMutations(Node node) {
        return node == this.sibling ? this.joinedMutations : this.mutations.getMutations(node);
    }

    /** Returns the given child in the pruned tree, where S takes the place of P. X is never a child of a junction, as P is none. */
    private Node getPrunedChild(Node child) {
        return child == this.x.getParent() ? this.sibling : child;
    }

    /**
     * Counts the sites whose states differ between the start and the end of the given
     * changes, leaving out the missing sites of X.
     */
    private int countObservedDifferences(SiteChanges changes) {
        int numDifferences = 0;
        for (int slot = 0; slot < changes.getSize(); slot++) {
            if (changes.getStartState(slot) != changes.getEndState(slot) && !this.subtreeMissingSiteSet.get(changes.getSite(slot))) {
                numDifferences++;
            }
        }
        return numDifferences;
    }

    /**
     * Returns how much crossing the given mutation downwards changes the number of
     * differences to X. The missing sites of X are left out, as the move resamples the
     * states of X there, and the grafting density must not depend on them.
     */
    private int computeDifferenceChange(Mutation mutation) {
        if (this.subtreeMissingSiteSet.get(mutation.site())) {
            return 0;
        }

        int subtreeState = this.getSubtreeState(mutation.site());
        return (mutation.newState() != subtreeState ? 1 : 0) - (mutation.oldState() != subtreeState ? 1 : 0);
    }

    /** Returns the state of X at the given site. */
    private int getSubtreeState(int site) {
        int slot = this.subtreeSequenceChanges.getSlot(site);
        return slot >= 0 ? this.subtreeSequenceChanges.getEndState(slot) : this.referenceSequence[site];
    }

    /* Grafting Density */

    /**
     * Turns the log weights in the region buffers into weights relative to the largest one
     * and returns the log of their total.
     */
    private double normaliseRegionWeights() {
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < this.numRegions; i++) {
            max = Math.max(max, this.regionWeights[i]);
        }

        double sum = 0.0;
        for (int i = 0; i < this.numRegions; i++) {
            this.regionWeights[i] = Math.exp(this.regionWeights[i] - max);
            sum += this.regionWeights[i];
        }
        return max + Math.log(sum);
    }

    /** Samples a region with probability proportional to its weight. */
    private int sampleRegion() {
        double totalWeight = 0.0;
        for (int i = 0; i < this.numRegions; i++) {
            totalWeight += this.regionWeights[i];
        }

        double threshold = Randomizer.nextDouble() * totalWeight;
        double cumulativeWeight = 0.0;
        for (int i = 0; i < this.numRegions; i++) {
            cumulativeWeight += this.regionWeights[i];
            if (threshold < cumulativeWeight) {
                return i;
            }
        }

        // only reached through rounding, so return the last region with positive weight
        for (int i = this.numRegions - 1; i > 0; i--) {
            if (this.regionWeights[i] > 0.0) {
                return i;
            }
        }
        return 0;
    }

    /** Computes the log weight of a region: its length times the grafting density at its midpoint. */
    private double computeLogWeight(double lowerHeight, double upperHeight, int numDifferences) {
        return Math.log(upperHeight - lowerHeight) + this.computeLogDensity(lowerHeight, upperHeight, numDifferences);
    }

    private double computeLogDensity(Region region) {
        return this.computeLogDensity(region.lowerHeight(), region.upperHeight(), region.numDifferences());
    }

    /**
     * Computes the log grafting density log g at the midpoint of the given height range, i.e.
     * f (-λτ + N log(λτ / 3L)) with τ the distance of the midpoint above X.
     */
    private double computeLogDensity(double lowerHeight, double upperHeight, int numDifferences) {
        double distance = 0.5 * (lowerHeight + upperHeight) - this.x.getHeight();

        double logDensity = -this.subtreeTotalRate * distance
                + numDifferences * (this.logRatePerSite + Math.log(distance));
        return this.annealing * logDensity;
    }

    /* Move Distances */

    /**
     * Measures the distance between the old and the new attachment point in the pruned
     * tree, which is the same before and after the move, as the number of nodes and the
     * length of the path between them. The path leads through the most recent common
     * ancestor M of the old sibling S and the new sibling S', and the heights change
     * monotonically on the way from either point to M.
     */
    private void measureProposedMove() {
        Node oldSibling = this.sibling;
        Node newSibling = this.proposedGraftingPoint.newSibling();
        double newParentHeight = this.proposedGraftingPoint.newParentHeight();

        if (newSibling == oldSibling) {
            this.proposedNumNodes = 0;
            this.proposedPathLength = Math.abs(newParentHeight - this.oldParentHeight);
        } else {
            int steps = 0;
            for (Node node = oldSibling; node != null; node = this.getPrunedParent(this.x, node)) {
                this.ancestorSteps[node.getNr()] = steps++;
            }

            Node mrca = newSibling;
            int numNewSteps = 0;
            while (this.ancestorSteps[mrca.getNr()] < 0) {
                mrca = this.getPrunedParent(this.x, mrca);
                numNewSteps++;
            }
            int numOldSteps = this.ancestorSteps[mrca.getNr()];

            for (Node node = oldSibling; node != null; node = this.getPrunedParent(this.x, node)) {
                this.ancestorSteps[node.getNr()] = -1;
            }

            // M is crossed by the paths from both siblings unless it is one of them
            boolean isMrcaSibling = mrca == oldSibling || mrca == newSibling;
            this.proposedNumNodes = numOldSteps + numNewSteps - (isMrcaSibling ? 0 : 1);
            this.proposedPathLength = Math.abs(mrca.getHeight() - this.oldParentHeight) + Math.abs(mrca.getHeight() - newParentHeight);
        }

        this.numMeasuredProposals++;
        this.totalProposedNumNodes += this.proposedNumNodes;
        this.totalProposedPathLength += this.proposedPathLength;
        if (this.proposedNumNodes == 0) {
            this.numSameBranchProposals++;
        }
    }

    /** Records the distance of the accepted move for the diagnostics and the subtree selection. */
    @Override
    public void accept() {
        super.accept();
        if (this.proposedGraftingPoint == null) {
            return;
        }

        this.numMeasuredAccepts++;
        this.totalAcceptedNumNodes += this.proposedNumNodes;
        this.totalAcceptedPathLength += this.proposedPathLength;

        if (this.isMoveDistanceSelection()) {
            this.recordMove(this.getSelectionDistance());
        }
    }

    /** Records the rejected proposal as a move of distance zero, if rejected proposals count. */
    @Override
    public void reject(int reason) {
        super.reject(reason);
        if (this.pickedSubtreeRoot != null && this.countRejectedMoves && this.isMoveDistanceSelection()) {
            this.recordMove(0.0);
        }
    }

    private boolean isMoveDistanceSelection() {
        return this.subtreeSelection != SubtreeSelection.UNIFORM && this.subtreeSelection != SubtreeSelection.BRANCH_LENGTH;
    }

    /** Returns the distance of the proposed move in the measure of the subtree selection. */
    private double getSelectionDistance() {
        return switch (this.subtreeSelection) {
            case MOVED_NUM_NODES -> this.proposedNumNodes;
            case MOVED_PATH_LENGTH -> this.proposedPathLength;
            case MOVED_SQUARED_PATH_LENGTH -> this.proposedPathLength * this.proposedPathLength;
            case UNIFORM, BRANCH_LENGTH -> throw new IllegalStateException("No move distance for " + this.subtreeSelection + ".");
        };
    }

    /** Adds a move of the given distance to the running mean of the picked subtree root, once per proposal. */
    private void recordMove(double distance) {
        int nr = this.pickedSubtreeRoot.getNr();
        this.numMoves[nr]++;
        this.totalMoveDistances[nr] += distance;
        this.numAllMoves++;
        this.totalAllMoveDistance += distance;

        this.pickedSubtreeRoot = null;
        this.proposedGraftingPoint = null;
    }

    /* Diagnostics */

    /**
     * Returns the diagnostics of the proposals so far by name: the mean number of candidate
     * regions of the local explorations, the mean distances of the proposed and the accepted
     * moves, the fraction of proposals that stay on the joined branch, and how evenly the
     * subtrees are picked. The effective number of subtrees is (Σc)² / Σc² over the pick
     * counts c, or over the selection weights in the current state, and equals the number of
     * candidates if all are picked equally often.
     */
    public Map<String, Double> getDiagnostics() {
        Map<String, Double> diagnostics = new LinkedHashMap<>();
        diagnostics.put("mdspr_local_regions", this.getMeanNumLocalRegions());
        diagnostics.put("mdspr_mean_proposed_nodes", (double) this.totalProposedNumNodes / this.numMeasuredProposals);
        diagnostics.put("mdspr_mean_proposed_path_length", this.totalProposedPathLength / this.numMeasuredProposals);
        diagnostics.put("mdspr_same_branch_fraction", (double) this.numSameBranchProposals / this.numMeasuredProposals);
        diagnostics.put("mdspr_mean_accepted_nodes", (double) this.totalAcceptedNumNodes / this.numMeasuredAccepts);
        diagnostics.put("mdspr_mean_accepted_path_length", this.totalAcceptedPathLength / this.numMeasuredAccepts);

        double totalPicks = 0.0;
        double totalSquaredPicks = 0.0;
        double tipPicks = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            double picks = this.numPicks[node.getNr()];
            totalPicks += picks;
            totalSquaredPicks += picks * picks;
            if (node.isLeaf()) {
                tipPicks += picks;
            }
        }
        diagnostics.put("mdspr_effective_picked_subtrees", totalPicks * totalPicks / totalSquaredPicks);
        diagnostics.put("mdspr_tip_pick_fraction", tipPicks / totalPicks);

        double totalWeight = this.computeSelectionWeights();
        double totalSquaredWeight = 0.0;
        int numCandidates = 0;
        for (double weight : this.selectionWeights) {
            totalSquaredWeight += weight * weight;
            numCandidates += weight > 0.0 ? 1 : 0;
        }
        diagnostics.put("mdspr_effective_weighted_subtrees", totalWeight * totalWeight / totalSquaredWeight);
        diagnostics.put("mdspr_num_candidates", (double) numCandidates);
        return diagnostics;
    }

}
