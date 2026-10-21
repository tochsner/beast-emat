package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.helper.SiteChanges;
import emat.helper.TipNeighbours;
import emat.operators.mdspr.BranchLengthSubtreeSelector;
import emat.operators.mdspr.CarrierGroupSubtreeSelector;
import emat.operators.mdspr.MoveDistanceSubtreeSelector;
import emat.operators.mdspr.ProposedMove;
import emat.operators.mdspr.Region;
import emat.operators.mdspr.RegionBuffer;
import emat.operators.mdspr.SisterActSubtreeSelector;
import emat.operators.mdspr.SubtreeSelector;
import emat.operators.mdspr.UniformSubtreeSelector;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Description("Regrafts a random subtree preferably where the sequence is close to the sequence of the subtree, " +
        "using the mutations of the EMAT as a parsimony map (mutation-directed SPR, docs/mcmc-moves.md §8.3). " +
        "Most proposals only explore the neighbourhood of the current attachment point, whose extent is drawn " +
        "from a geometric distribution, and may jump between tips with close data. The root is never changed.")
public class MutationDirectedSprOperator extends SubtreePruneRegraftOperator {

    public enum SubtreeSelection {
        UNIFORM, BRANCH_LENGTH, MOVED_NUM_NODES, MOVED_PATH_LENGTH, MOVED_SQUARED_PATH_LENGTH, SISTER_ACT, CARRIER_GROUP
    }

    final public Input<Double> annealingInput = new Input<>("annealing", "the power f to which the grafting density is raised, which makes the proposals less eager", 0.8);
    final public Input<Double> fullExplorationProbabilityInput = new Input<>("fullExplorationProbability", "the probability of exploring the whole tree instead of the neighbourhood of the current attachment point", 0.01);
    final public Input<Double> windowProbabilityInput = new Input<>("windowProbability", "the parameter q of the geometric number of additional differences, q^j (1 - q), that the neighbourhood may have compared to the current attachment point. With 0, the neighbourhood never has more differences than the current attachment point", 0.0);

    // the subtree selection and the parameters of its strategies, which each strategy validates and uses on its own
    final public Input<SubtreeSelection> subtreeSelectionInput = new Input<>("subtreeSelection", "the strategy for choosing the subtree to regraft: UNIFORM picks every subtree with the same probability, " +
            "BRANCH_LENGTH proportional to the length of the branch above it, and MOVED_NUM_NODES, MOVED_PATH_LENGTH and MOVED_SQUARED_PATH_LENGTH " +
            "proportional to the running mean distance of its accepted moves, measured as the number of nodes, the length or the squared length " +
            "of the path between the old and the new attachment point. SISTER_ACT only picks tips, in proportion to a power of the autocorrelation time of the size of their sister clade, " +
            "which is estimated from the history of the chain, so that tips that rarely change their sister are picked more often. " +
            "CARRIER_GROUP picks the groups of tips that switch between sitting apart from and sitting with the other carriers of a state at a homoplastic site, " +
            "which are learned from the history of the chain, mixed with picks in proportion to the branch length", SubtreeSelection.UNIFORM, SubtreeSelection.values());
    final public Input<Double> moveDistancePseudocountInput = new Input<>("moveDistancePseudocount", "the number of pseudo-moves with the mean distance of all recorded moves " +
            "that the running mean distance of every subtree starts with", 10.0);
    final public Input<Boolean> countRejectedMovesInput = new Input<>("countRejectedMoves", "whether the running mean distance of a subtree also counts its rejected proposals " +
            "as moves of distance zero, which makes it the expected distance per proposal. Otherwise, it is the mean distance of the accepted moves", false);
    final public Input<Double> sisterActPowerInput = new Input<>("sisterActPower", "the power to which the estimated autocorrelation time of the sister clade size of a tip is raised for SISTER_ACT. " +
            "With 0, all tips are picked equally often; larger values concentrate the picks on the slowest tips", 3.0);
    final public Input<Double> sisterActFloorInput = new Input<>("sisterActFloor", "the share of the SISTER_ACT picks that is spread uniformly over all tips, so that every tip keeps being picked", 0.1);
    final public Input<Integer> sisterActIntervalInput = new Input<>("sisterActInterval", "the number of proposals between two samples of the sister clade sizes of all tips for SISTER_ACT", 100);
    final public Input<Integer> sisterActWarmupInput = new Input<>("sisterActWarmup", "the number of samples of the sister clade sizes that SISTER_ACT discards before it starts estimating, while all tips are picked equally often", 1000);
    final public Input<Double> carrierGroupPowerInput = new Input<>("carrierGroupPower", "the power to which the estimated autocorrelation time of a carrier group is raised for CARRIER_GROUP. " +
            "With 0, all groups that switch are picked equally often", 1.0);
    final public Input<Double> carrierGroupFloorInput = new Input<>("carrierGroupFloor", "the share of the CARRIER_GROUP picks in proportion to the branch length, so that every subtree keeps being picked. " +
            "With 0, only carrier groups are picked, and the proposal is rejected while none can be picked, including before the weights are frozen", 0.8);
    final public Input<Double> carrierGroupMinFrequencyInput = new Input<>("carrierGroupMinFrequency", "the smallest share of the samples in which a carrier group must sit apart, " +
            "and with the other carriers, to be picked by CARRIER_GROUP", 0.01);
    final public Input<Integer> carrierGroupIntervalInput = new Input<>("carrierGroupInterval", "the number of proposals between two samples of the carrier groups for CARRIER_GROUP", 100);
    final public Input<Integer> carrierGroupBurninInput = new Input<>("carrierGroupBurnin", "the number of samples of the carrier groups that CARRIER_GROUP discards before it starts learning", 1000);
    final public Input<Integer> carrierGroupLearningSamplesInput = new Input<>("carrierGroupLearningSamples", "the number of samples of the carrier groups that CARRIER_GROUP learns its weights from before it freezes them, " +
            "while all subtrees are picked in proportion to the branch length", 2000);

    final public Input<Integer> shortcutDistanceInput = new Input<>("shortcutDistance", "the largest number of sites at which the data of two tips may differ for the neighbourhood " +
            "to also connect the regions at the bottom of their branches, as if a path ran between them on which the differences to the subtree never exceed the larger number of either end. " +
            "With a negative value, there are no such shortcuts", -1);
    final public Input<Boolean> scaleWeightByLogNumTipsInput = new Input<>("scaleWeightByLogNumTips", "whether to scale the weight by the natural logarithm of the number of tips n, so that the operator is selected with weight * ln(n). Otherwise, the weight is used as given", false);

    private static final int INITIAL_CAPACITY = 64;

    // the factor ln(n) or one by which the weight of the operator is scaled
    double weightFactor;

    double annealing;
    double fullExplorationProbability;
    double windowProbability;

    SubtreeSelector subtreeSelector;

    // the state of the current proposal: X, the joined mutations of the branch above S, and the constant part log(λ(X) / 3L) of the log grafting density
    Node x;
    Node sibling;
    List<Mutation> joinedMutations;
    double logRatePerSite;

    // the sequence of X as its differences to the reference sequence, which hold the state of X as end state
    SiteChanges subtreeSequenceChanges;
    int[] referenceSequence;

    // the explored regions of the current proposal
    final RegionBuffer regions = new RegionBuffer();

    // the junctions still to explore, i.e. nodes of the pruned tree where branches meet, with their number of differences to X
    int numJunctions;
    Node[] junctionNodes = new Node[INITIAL_CAPACITY];
    int[] junctionNumDifferences = new int[INITIAL_CAPACITY];

    // per node, the exploration that walked the run of regions at the top or the bottom of its branch
    int explorationStamp;
    int[] topRunStamps;
    int[] bottomRunStamps;
    int threshold;

    // per tip, the tips with close data that the bottom region of its branch has shortcuts to, or null without shortcuts
    TipNeighbours tipNeighbours;

    // per class of tips with identical data, the exploration that took the shortcuts of the class
    int[] classStamps;

    // per node, the exploration that checked the tip as the end of a shortcut, and the exploration that knows the number of differences at the node, with that number
    int[] shortcutStamps;
    int[] knownDifferenceStamps;
    int[] knownNumDifferences;

    // the nodes on the path from a tip up to a node with a known number of differences, with the change of the number from each of them down to the tip
    Node[] pathNodes;
    int[] pathDifferenceChanges;

    // the sites where X differs from the base sequence of the tip neighbours, which are collected once per exploration that needs them
    int[] subtreeBaseDifferenceSites;
    int numSubtreeBaseDifferences;
    boolean hasSubtreeBaseDifferences;

    // the subtree root of the current proposal, its old attachment point, its new grafting point and the distance between them, which are recorded as a move once accepted
    Node pickedSubtreeRoot;
    double oldParentHeight;
    GraftingPoint proposedGraftingPoint;
    ProposedMove proposedMove;

    // per node, the number of steps from the old sibling to the node, or -1 if it is no ancestor of the old sibling in the pruned tree
    int[] ancestorSteps;

    // the diagnostics of the proposals so far: the distances of the proposed and the accepted moves
    long numMeasuredProposals;
    long numSameBranchProposals;
    long totalProposedNumNodes;
    double totalProposedPathLength;
    long numMeasuredAccepts;
    long totalAcceptedNumNodes;
    double totalAcceptedPathLength;

    // the number of local explorations so far, the number of regions they explored and the number of tips they entered through shortcuts
    long numLocalExplorations;
    long numLocalRegions;
    long numShortcutTips;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.annealing = this.annealingInput.get();
        this.fullExplorationProbability = this.fullExplorationProbabilityInput.get();
        this.windowProbability = this.windowProbabilityInput.get();
        if (this.windowProbability < 0.0 || this.windowProbability >= 1.0) {
            throw new IllegalArgumentException("windowProbability must be at least 0 and less than 1.");
        }
        this.subtreeSelector = this.createSubtreeSelector();

        this.subtreeSequenceChanges = new SiteChanges(this.mutations.getReferenceSequence().length);

        int shortcutDistance = this.shortcutDistanceInput.get();
        if (shortcutDistance >= 0) {
            this.tipNeighbours = new TipNeighbours(this.mutations, this.tree, shortcutDistance);
            this.subtreeBaseDifferenceSites = new int[this.mutations.getReferenceSequence().length];
        }

        int numNodes = this.tree.getNodeCount();
        if (this.tipNeighbours != null) {
            this.classStamps = new int[this.tipNeighbours.getNumClasses()];
            this.shortcutStamps = new int[numNodes];
            this.knownDifferenceStamps = new int[numNodes];
            this.knownNumDifferences = new int[numNodes];
            this.pathNodes = new Node[numNodes];
            this.pathDifferenceChanges = new int[numNodes];
        }
        this.topRunStamps = new int[numNodes];
        this.bottomRunStamps = new int[numNodes];
        this.ancestorSteps = new int[numNodes];
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

    private SubtreeSelector createSubtreeSelector() {
        return switch (this.subtreeSelectionInput.get()) {
            case UNIFORM -> new UniformSubtreeSelector(this.tree);
            case BRANCH_LENGTH -> new BranchLengthSubtreeSelector(this.tree);
            case MOVED_NUM_NODES -> this.createMoveDistanceSelector(MoveDistanceSubtreeSelector.Measure.NUM_NODES);
            case MOVED_PATH_LENGTH -> this.createMoveDistanceSelector(MoveDistanceSubtreeSelector.Measure.PATH_LENGTH);
            case MOVED_SQUARED_PATH_LENGTH -> this.createMoveDistanceSelector(MoveDistanceSubtreeSelector.Measure.SQUARED_PATH_LENGTH);
            case SISTER_ACT -> new SisterActSubtreeSelector(
                    this.tree, this.sisterActPowerInput.get(), this.sisterActFloorInput.get(),
                    this.sisterActIntervalInput.get(), this.sisterActWarmupInput.get()
            );
            case CARRIER_GROUP -> new CarrierGroupSubtreeSelector(
                    this.tree, this.mutations, this.carrierGroupPowerInput.get(), this.carrierGroupFloorInput.get(), this.carrierGroupMinFrequencyInput.get(),
                    this.carrierGroupIntervalInput.get(), this.carrierGroupBurninInput.get(), this.carrierGroupLearningSamplesInput.get()
            );
        };
    }

    private SubtreeSelector createMoveDistanceSelector(MoveDistanceSubtreeSelector.Measure measure) {
        return new MoveDistanceSubtreeSelector(this.tree, measure, this.moveDistancePseudocountInput.get(), this.countRejectedMovesInput.get());
    }

    /** Picks X with the subtree selector among all nodes where neither X nor its parent is the root. */
    @Override
    protected Node pickSubtreeRoot() {
        this.proposedGraftingPoint = null;
        this.subtreeSelector.prepare();
        this.pickedSubtreeRoot = this.subtreeSelector.pick();
        return this.pickedSubtreeRoot;
    }

    /** Returns the log of the selection probability of X in the current state, which the move can change through the weights of the nodes around it. */
    @Override
    protected double computeLogSubtreeRootProbability(Node x) {
        return this.subtreeSelector.computeLogProbability(x);
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
     * <p>
     * With shortcuts, the regions at the bottom of the branches above tips with close data
     * are connected as well if both are within T. This is symmetric and does not depend on
     * the attachment point of X, so the reverse move still finds the same regions.
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
            this.numLocalRegions += this.regions.size();
        }
        double logTotalWeight = this.regions.normaliseWeights();

        Region newRegion = this.regions.sample();
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
     * <p>
     * Tips whose bottom region was added are pushed onto the stack as well, which then
     * continues through their shortcuts.
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

            if (junction.isLeaf()) {
                this.takeShortcuts(junction);
                continue;
            }

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
        this.regions.clear();
        this.numJunctions = 0;
        this.hasSubtreeBaseDifferences = false;

        // the stamps only need resetting once the counter runs out
        if (this.explorationStamp == Integer.MAX_VALUE) {
            Arrays.fill(this.topRunStamps, 0);
            Arrays.fill(this.bottomRunStamps, 0);
            if (this.tipNeighbours != null) {
                Arrays.fill(this.classStamps, 0);
                Arrays.fill(this.shortcutStamps, 0);
                Arrays.fill(this.knownDifferenceStamps, 0);
            }
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
            if (node.isLeaf()) {
                this.pushJunction(node, numDifferences);
            }
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

        // junctions not older than X only lead to regions below X, but tips may lead to shortcuts
        if (node.isLeaf() || node.getHeight() > this.x.getHeight()) {
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

        this.regions.add(
                new Region(node, index, lowerHeight, upperHeight, numDifferences),
                this.computeLogWeight(lowerHeight, upperHeight, numDifferences)
        );
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

        // the counts of the shortcuts start from the nearest node with a known number
        if (this.tipNeighbours != null) {
            this.recordNumDifferences(node, numDifferences);
        }
    }

    private void recordNumDifferences(Node node, int numDifferences) {
        this.knownDifferenceStamps[node.getNr()] = this.explorationStamp;
        this.knownNumDifferences[node.getNr()] = numDifferences;
    }

    /**
     * Walks the runs at the bottom of the branches above the tips with close data to the
     * given tip, whose bottom region was added. A shortcut stands for a path that first
     * reverts the sites where one tip differs from X and then introduces the sites where the
     * other one does, whose regions never have more differences than the larger number of
     * either end. Hence, it connects both bottom regions exactly if both are within the
     * threshold. A full exploration reaches every region anyway.
     * <p>
     * Whether a neighbour is entered only depends on the neighbour, so every tip is checked
     * at most once per exploration, however many of its neighbours are reached.
     */
    private void takeShortcuts(Node tip) {
        if (this.tipNeighbours == null || this.threshold == Integer.MAX_VALUE) {
            return;
        }

        // all tips of a class have the same neighbours, so the first of them that is reached takes them all
        int classNr = this.tipNeighbours.getClassNr(tip);
        if (this.classStamps[classNr] == this.explorationStamp) {
            return;
        }
        this.classStamps[classNr] = this.explorationStamp;

        this.enterClassFromBottom(classNr);
        for (int neighbourClassNr : this.tipNeighbours.getNeighbourClasses(classNr)) {
            this.enterClassFromBottom(neighbourClassNr);
        }
    }

    private void enterClassFromBottom(int classNr) {
        for (int nr : this.tipNeighbours.getMembers(classNr)) {
            if (this.shortcutStamps[nr] == this.explorationStamp) {
                continue;
            }
            this.shortcutStamps[nr] = this.explorationStamp;

            Node neighbour = this.tree.getNode(nr);
            if (neighbour == this.x || this.bottomRunStamps[nr] == this.explorationStamp) {
                continue;
            }

            // tips inside X have no bottom region older than X, as the branches above them end below X
            int index = this.getBranchMutations(neighbour).size();
            if (this.getUpperHeight(neighbour, index) <= this.getLowerHeight(neighbour, index)) {
                continue;
            }

            // the data of the neighbour bound its number of differences from below, which spares counting most neighbours beyond the threshold
            if (this.countDefiniteDifferences(neighbour) > this.threshold) {
                continue;
            }

            this.numShortcutTips++;
            this.enterFromBottom(neighbour, this.countTipDifferences(neighbour));
        }
    }

    /**
     * Counts the differences between the sequence of the given tip and X at the sites where
     * the data of the tip is a single state, which fixes the state of the tip there. This
     * is a lower bound of the differences of the region at the bottom of the branch above
     * the tip, which may also differ where the data of the tip is missing or ambiguous.
     */
    private int countDefiniteDifferences(Node tip) {
        if (!this.hasSubtreeBaseDifferences) {
            this.collectSubtreeBaseDifferences();
        }

        int[] definiteSites = this.tipNeighbours.getDefiniteSites(tip);
        int[] definiteStates = this.tipNeighbours.getDefiniteStates(tip);

        int numDifferences = 0;
        for (int i = 0; i < definiteSites.length; i++) {
            numDifferences += this.isDifferentState(definiteSites[i], definiteStates[i]) ? 1 : 0;
        }

        // at the other sites where X differs from the base, a definite tip has the base state
        BitSet tipMissingSites = this.missingSites.getSiteSet(tip);
        BitSet tipAmbiguousSites = this.missingSites.getAmbiguousSiteSet(tip);
        for (int i = 0; i < this.numSubtreeBaseDifferences; i++) {
            int site = this.subtreeBaseDifferenceSites[i];
            if (!tipMissingSites.get(site) && !tipAmbiguousSites.get(site) && Arrays.binarySearch(definiteSites, site) < 0) {
                numDifferences++;
            }
        }
        return numDifferences;
    }

    /**
     * Collects the sites where the base state counts as a difference to X. X can only differ
     * from the base where its sequence differs from the reference sequence or the base does.
     */
    private void collectSubtreeBaseDifferences() {
        int[] baseSequence = this.tipNeighbours.getBaseSequence();
        this.numSubtreeBaseDifferences = 0;

        for (int slot = 0; slot < this.subtreeSequenceChanges.getSize(); slot++) {
            int site = this.subtreeSequenceChanges.getSite(slot);
            if (this.isDifferentState(site, baseSequence[site])) {
                this.subtreeBaseDifferenceSites[this.numSubtreeBaseDifferences++] = site;
            }
        }
        for (int site : this.tipNeighbours.getBaseChangeSites()) {
            if (!this.subtreeSequenceChanges.containsSite(site) && this.isDifferentState(site, baseSequence[site])) {
                this.subtreeBaseDifferenceSites[this.numSubtreeBaseDifferences++] = site;
            }
        }
        this.hasSubtreeBaseDifferences = true;
    }

    /**
     * Counts the differences between the sequence of the given tip and X in the same way as
     * the walks, so that it matches the number of the region at the bottom of the branch
     * above the tip. It walks up from the tip to the nearest node whose number is known in
     * this exploration, or to the top of the root branch, where the sequence is the
     * reference sequence, and adds the changes of the crossed mutations. The numbers of
     * the nodes on the way become known as well, so later counts stop there.
     */
    private int countTipDifferences(Node tip) {
        int numPathNodes = 0;
        int differenceChange = 0;
        Node node = tip;
        while (node != null && this.knownDifferenceStamps[node.getNr()] != this.explorationStamp) {
            this.pathNodes[numPathNodes] = node;
            this.pathDifferenceChanges[numPathNodes] = differenceChange;
            numPathNodes++;

            for (Mutation mutation : this.getBranchMutations(node)) {
                differenceChange += this.computeDifferenceChange(mutation);
            }
            node = node.isRoot() ? null : this.getPrunedParent(this.x, node);
        }

        int numDifferences = differenceChange + (node == null ? this.countReferenceDifferences() : this.knownNumDifferences[node.getNr()]);
        for (int i = 0; i < numPathNodes; i++) {
            this.recordNumDifferences(this.pathNodes[i], numDifferences - this.pathDifferenceChanges[i]);
        }
        return numDifferences;
    }

    /**
     * Counts the differences between the reference sequence and X. X only differs from it
     * where its sequence changes, as its states at the resampled ambiguous sites are
     * compatible with its data.
     */
    private int countReferenceDifferences() {
        int numDifferences = 0;
        for (int slot = 0; slot < this.subtreeSequenceChanges.getSize(); slot++) {
            int site = this.subtreeSequenceChanges.getSite(slot);
            numDifferences += this.isDifferentState(site, this.referenceSequence[site]) ? 1 : 0;
        }
        return numDifferences;
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
     * changes, leaving out the missing sites of X. At the resampled ambiguous sites of X,
     * the start state only counts if it is incompatible with the data of X.
     */
    private int countObservedDifferences(SiteChanges changes) {
        int numDifferences = 0;
        for (int slot = 0; slot < changes.getSize(); slot++) {
            int site = changes.getSite(slot);
            if (changes.getStartState(slot) == changes.getEndState(slot) || this.subtreeMissingSiteSet.get(site)) {
                continue;
            }

            numDifferences += !this.subtreeAmbiguousSiteSet.get(site) || !this.isCompatibleState(site, changes.getStartState(slot)) ? 1 : 0;
        }
        return numDifferences;
    }

    /**
     * Returns how much crossing the given mutation downwards changes the number of
     * differences to X. The missing sites of X are left out, as the move resamples the
     * states of X there, and the grafting density must not depend on them. At the
     * resampled ambiguous sites of X, a state differs only if it is incompatible with the
     * data of X, which does not depend on the current states of X there either.
     */
    private int computeDifferenceChange(Mutation mutation) {
        int site = mutation.site();
        return (this.isDifferentState(site, mutation.newState()) ? 1 : 0) - (this.isDifferentState(site, mutation.oldState()) ? 1 : 0);
    }

    /** Checks whether the given state counts as a difference to X at the given site. */
    private boolean isDifferentState(int site, int state) {
        if (this.subtreeMissingSiteSet.get(site)) {
            return false;
        }
        if (this.subtreeAmbiguousSiteSet.get(site)) {
            return !this.isCompatibleState(site, state);
        }
        return state != this.getSubtreeState(site);
    }

    /** Checks whether the given state is compatible with the data of X at the given resampled ambiguous site. */
    private boolean isCompatibleState(int site, int state) {
        int index = Arrays.binarySearch(this.subtreeAmbiguousSites, site);
        return (this.subtreeCompatibleStates[index] & (1 << state)) != 0;
    }

    /** Returns the state of X at the given site. */
    private int getSubtreeState(int site) {
        int slot = this.subtreeSequenceChanges.getSlot(site);
        return slot >= 0 ? this.subtreeSequenceChanges.getEndState(slot) : this.referenceSequence[site];
    }

    /* Grafting Density */

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
            this.proposedMove = new ProposedMove(0, Math.abs(newParentHeight - this.oldParentHeight));
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
            this.proposedMove = new ProposedMove(
                    numOldSteps + numNewSteps - (isMrcaSibling ? 0 : 1),
                    Math.abs(mrca.getHeight() - this.oldParentHeight) + Math.abs(mrca.getHeight() - newParentHeight)
            );
        }

        this.numMeasuredProposals++;
        this.totalProposedNumNodes += this.proposedMove.numNodes();
        this.totalProposedPathLength += this.proposedMove.pathLength();
        if (this.proposedMove.numNodes() == 0) {
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
        this.totalAcceptedNumNodes += this.proposedMove.numNodes();
        this.totalAcceptedPathLength += this.proposedMove.pathLength();

        this.subtreeSelector.acceptMove(this.pickedSubtreeRoot, this.proposedMove);
        this.pickedSubtreeRoot = null;
        this.proposedGraftingPoint = null;
    }

    /** Lets the subtree selection learn from the rejected proposal. */
    @Override
    public void reject(int reason) {
        super.reject(reason);
        if (this.pickedSubtreeRoot != null) {
            this.subtreeSelector.rejectMove(this.pickedSubtreeRoot);
            this.pickedSubtreeRoot = null;
        }
    }

    /* Diagnostics */

    /**
     * Returns the diagnostics of the proposals so far by name: the mean number of candidate
     * regions of the local explorations, the mean distances of the proposed and the accepted
     * moves, the fraction of proposals that stay on the joined branch, and the diagnostics
     * of the subtree selection.
     */
    public Map<String, Double> getDiagnostics() {
        Map<String, Double> diagnostics = new LinkedHashMap<>();
        diagnostics.put("mdspr_local_regions", this.getMeanNumLocalRegions());
        diagnostics.put("mdspr_shortcut_tips", this.numLocalExplorations == 0 ? Double.NaN : (double) this.numShortcutTips / this.numLocalExplorations);
        diagnostics.put("mdspr_mean_proposed_nodes", (double) this.totalProposedNumNodes / this.numMeasuredProposals);
        diagnostics.put("mdspr_mean_proposed_path_length", this.totalProposedPathLength / this.numMeasuredProposals);
        diagnostics.put("mdspr_same_branch_fraction", (double) this.numSameBranchProposals / this.numMeasuredProposals);
        diagnostics.put("mdspr_mean_accepted_nodes", (double) this.totalAcceptedNumNodes / this.numMeasuredAccepts);
        diagnostics.put("mdspr_mean_accepted_path_length", this.totalAcceptedPathLength / this.numMeasuredAccepts);

        this.subtreeSelector.addDiagnostics(diagnostics);
        diagnostics.put("mdspr_sister_act_samples", (double) this.subtreeSelector.getNumSamples());
        return diagnostics;
    }

}
