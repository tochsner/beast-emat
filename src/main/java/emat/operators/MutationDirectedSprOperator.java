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
import java.util.List;

@Description("Regrafts a random subtree preferably where the sequence is close to the sequence of the subtree, " +
        "using the mutations of the EMAT as a parsimony map (mutation-directed SPR, docs/mcmc-moves.md §8.3). " +
        "Most proposals only explore the neighbourhood of the current attachment point. The root is never changed.")
public class MutationDirectedSprOperator extends SubtreePruneRegraftOperator {

    final public Input<Double> annealingInput = new Input<>("annealing", "the power f to which the grafting density is raised, which makes the proposals less eager", 0.8);
    final public Input<Double> fullExplorationProbabilityInput = new Input<>("fullExplorationProbability", "the probability of exploring the whole tree instead of the neighbourhood of the current attachment point", 0.01);

    private static final int INITIAL_CAPACITY = 64;

    double annealing;
    double fullExplorationProbability;

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

        this.annealing = this.annealingInput.get();
        this.fullExplorationProbability = this.fullExplorationProbabilityInput.get();

        this.subtreeSequenceChanges = new SiteChanges(this.mutations.getReferenceSequence().length);

        int numNodes = this.tree.getNodeCount();
        this.topRunStamps = new int[numNodes];
        this.bottomRunStamps = new int[numNodes];
    }

    /**
     * Picks X uniformly among all nodes where neither X nor its parent is the root. This set
     * does not change under moves that keep the root, so the choice is symmetric.
     */
    @Override
    protected Node pickSubtreeRoot() {
        List<Node> candidates = new ArrayList<>();
        for (Node node : this.tree.getNodesAsArray()) {
            if (!node.isRoot() && !node.getParent().isRoot()) {
                candidates.add(node);
            }
        }

        if (candidates.isEmpty()) {
            return null;
        }
        return candidates.get(Randomizer.nextInt(candidates.size()));
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
     * regions connected to the current attachment point through regions with at most
     * max(1, N) differences are candidates, where N is the number of differences at the
     * current attachment point. The reverse move explores from the new attachment point
     * with its own threshold. If that threshold is the same, it finds the same regions, so
     * the total weight cancels as well. If it is lower, it cannot reach the current
     * attachment point, whose number of differences exceeds it, and the move is rejected.
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
        int threshold = isFullExploration ? Integer.MAX_VALUE : Math.max(1, oldNumDifferences);

        this.exploreRegions(oldRegion, threshold);
        double logTotalWeight = this.normaliseRegionWeights();

        int newRegionNr = this.sampleRegion();
        Region newRegion = this.getRegion(newRegionNr);
        double newParentHeight = newRegion.lowerHeight() + Randomizer.nextDouble() * (newRegion.upperHeight() - newRegion.lowerHeight());

        if (!isFullExploration && Math.max(1, newRegion.numDifferences()) < threshold) {
            // the reverse move cannot propose the current attachment point
            return null;
        }

        // the density of a point in a region is g(midpoint) divided by the total weight of the candidates

        double logForwardDensity = this.computeLogDensity(newRegion) - logTotalWeight;
        double logBackwardDensity = this.computeLogDensity(oldRegion) - logTotalWeight;

        return new GraftingPoint(newRegion.node(), newParentHeight, logBackwardDensity - logForwardDensity);
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

}
