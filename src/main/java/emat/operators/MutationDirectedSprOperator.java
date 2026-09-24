package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.helper.StochasticMapping;
import emat.state.Mutation;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Description("Regrafts a random subtree preferably where the sequence is close to the sequence of the subtree, " +
        "using the mutations of the EMAT as a parsimony map (mutation-directed SPR, docs/mcmc-moves.md §8.3). " +
        "Most proposals only explore the neighbourhood of the current attachment point. The root is never changed.")
public class MutationDirectedSprOperator extends SubtreePruneRegraftOperator {

    final public Input<Double> annealingInput = new Input<>("annealing", "the power f to which the grafting density is raised, which makes the proposals less eager", 0.8);
    final public Input<Double> fullExplorationProbabilityInput = new Input<>("fullExplorationProbability", "the probability of exploring the whole tree instead of the neighbourhood of the current attachment point", 0.01);

    double annealing;
    double fullExplorationProbability;

    // the state of the current proposal: X, its sequence, its total mutation rate, and the joined mutations of the branch above S
    Node x;
    int[] subtreeSequence;
    double totalRate;
    Node sibling;
    List<Mutation> joinedMutations;

    /**
     * A part of a branch of the pruned tree with a constant sequence: the node below the
     * branch, the index of the part counted from the top of the branch, the height range
     * of the part that is older than X, and the number of sites where its sequence differs
     * from the sequence of X.
     */
    record Region(Node node, int index, double lowerHeight, double upperHeight, int numDifferences) {
    }

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.annealing = this.annealingInput.get();
        this.fullExplorationProbability = this.fullExplorationProbabilityInput.get();
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
     * Jukes–Cantor with rate λ(X) spread over L sites, the density of attaching at distance
     * τ above X to a sequence with N differences is g = [exp(-λτ) (λτ / 3L)^N]^f (Eq. 44).
     * <p>
     * With a small fixed probability, all regions of the pruned tree are candidates. Their
     * total weight is the same before and after the move, so it cancels. Otherwise, only the
     * regions connected to the current attachment point through regions with at most
     * max(1, N) differences are candidates, where N is the number of differences at the
     * current attachment point. The reverse move explores from the new attachment point
     * with its own threshold, and the move is rejected if that does not reach the old one.
     */
    @Override
    protected GraftingPoint proposeGraftingPoint(Node x) {
        this.x = x;
        this.subtreeSequence = MutationPaths.getSequence(this.mutations, x);
        this.totalRate = this.computeTotalRate(this.subtreeSequence);

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

        int oldNumDifferences = MutationPaths.combineChanges(
                new HashMap<>(), MutationPaths.collectChanges(this.mutations, parent, x, x.getHeight())
        ).size();

        Region oldRegion = this.createRegion(this.sibling, oldIndex, oldNumDifferences);

        // explore the candidate regions and pick one

        boolean isFullExploration = Randomizer.nextDouble() < this.fullExplorationProbability;
        int threshold = isFullExploration ? Integer.MAX_VALUE : Math.max(1, oldNumDifferences);

        List<Region> regions = this.exploreRegions(oldRegion, threshold);
        double[] logWeights = this.computeLogWeights(regions);
        double logTotalWeight = this.computeLogSum(logWeights);

        double[] weights = new double[regions.size()];
        for (int i = 0; i < regions.size(); i++) {
            weights[i] = Math.exp(logWeights[i] - logTotalWeight);
        }

        Region newRegion = regions.get(StochasticMapping.sampleIndex(weights));
        double newParentHeight = newRegion.lowerHeight() + Randomizer.nextDouble() * (newRegion.upperHeight() - newRegion.lowerHeight());

        // the density of a point in a region is g(midpoint) divided by the total weight of the candidates

        double logForwardDensity = this.computeLogDensity(newRegion) - logTotalWeight;
        double logBackwardDensity;

        if (isFullExploration) {
            logBackwardDensity = this.computeLogDensity(oldRegion) - logTotalWeight;
        } else {
            List<Region> reverseRegions = this.exploreRegions(newRegion, Math.max(1, newRegion.numDifferences()));
            if (!this.containsRegion(reverseRegions, oldRegion)) {
                // the reverse move cannot propose the current attachment point
                return null;
            }
            logBackwardDensity = this.computeLogDensity(oldRegion) - this.computeLogSum(this.computeLogWeights(reverseRegions));
        }

        return new GraftingPoint(newRegion.node(), newParentHeight, logBackwardDensity - logForwardDensity);
    }

    /* Regions */

    /**
     * Collects the regions older than X that are connected to the given region through
     * regions with at most the given number of differences. Crossing a mutation changes
     * the number of differences at its site, and crossing a node keeps it.
     */
    private List<Region> exploreRegions(Region start, int threshold) {
        List<Region> regions = new ArrayList<>();
        Set<Long> visited = new HashSet<>();
        Deque<Region> stack = new ArrayDeque<>();
        stack.push(start);

        while (!stack.isEmpty()) {
            Region region = stack.pop();
            if (!visited.add(this.getRegionKey(region.node(), region.index()))) {
                continue;
            }
            regions.add(region);

            Node node = region.node();
            List<Mutation> branchMutations = this.getBranchMutations(node);
            int index = region.index();
            int numDifferences = region.numDifferences();

            // move up, across the mutation above or into the branches that meet at the top of the branch

            if (index > 0) {
                Mutation mutation = branchMutations.get(index - 1);
                this.pushRegion(stack, node, index - 1, numDifferences - this.computeDifferenceChange(mutation), threshold);
            } else {
                Node top = this.getPrunedParent(this.x, node);
                for (Node child : this.getPrunedChildren(top)) {
                    if (child != node) {
                        this.pushRegion(stack, child, 0, numDifferences, threshold);
                    }
                }
                if (!top.isRoot()) {
                    this.pushRegion(stack, top, this.getBranchMutations(top).size(), numDifferences, threshold);
                }
            }

            // move down, across the mutation below or into the branches below the node

            if (index < branchMutations.size()) {
                Mutation mutation = branchMutations.get(index);
                this.pushRegion(stack, node, index + 1, numDifferences + this.computeDifferenceChange(mutation), threshold);
            } else {
                for (Node child : this.getPrunedChildren(node)) {
                    this.pushRegion(stack, child, 0, numDifferences, threshold);
                }
            }
        }

        return regions;
    }

    /**
     * Pushes the given region onto the stack if it is within the threshold and has a part
     * older than X. Regions below X only lead further down, so they can be skipped.
     */
    private void pushRegion(Deque<Region> stack, Node node, int index, int numDifferences, int threshold) {
        if (numDifferences > threshold) {
            return;
        }

        Region region = this.createRegion(node, index, numDifferences);
        if (region.upperHeight() > region.lowerHeight()) {
            stack.push(region);
        }
    }

    /**
     * Creates the region with the given index on the branch above the given node, from the
     * mutation above it (or the top of the branch) down to the mutation below it (or the
     * node), truncated at the height of X.
     */
    private Region createRegion(Node node, int index, int numDifferences) {
        List<Mutation> branchMutations = this.getBranchMutations(node);

        double upperHeight = index == 0
                ? this.getPrunedParent(this.x, node).getHeight()
                : branchMutations.get(index - 1).time();
        double lowerHeight = index == branchMutations.size()
                ? node.getHeight()
                : branchMutations.get(index).time();

        return new Region(node, index, Math.max(lowerHeight, this.x.getHeight()), upperHeight, numDifferences);
    }

    /** Returns the mutations on the branch above the given node in the pruned tree. */
    private List<Mutation> getBranchMutations(Node node) {
        return node == this.sibling ? this.joinedMutations : this.mutations.getMutations(node);
    }

    /** Returns the children of the given node in the pruned tree, where S takes the place of P and X is removed. */
    private List<Node> getPrunedChildren(Node node) {
        Node parent = this.x.getParent();

        List<Node> children = new ArrayList<>();
        for (Node child : node.getChildren()) {
            if (child == parent) {
                children.add(this.sibling);
            } else if (child != this.x) {
                children.add(child);
            }
        }
        return children;
    }

    /** Returns how much crossing the given mutation downwards changes the number of differences to X. */
    private int computeDifferenceChange(Mutation mutation) {
        int subtreeState = this.subtreeSequence[mutation.site()];
        return (mutation.newState() != subtreeState ? 1 : 0) - (mutation.oldState() != subtreeState ? 1 : 0);
    }

    private long getRegionKey(Node node, int index) {
        return ((long) node.getNr() << 32) | index;
    }

    private boolean containsRegion(List<Region> regions, Region region) {
        for (Region candidate : regions) {
            if (candidate.node() == region.node() && candidate.index() == region.index()) {
                return true;
            }
        }
        return false;
    }

    /* Grafting Density */

    /** Computes the log weight of every region: its length times the grafting density at its midpoint. */
    private double[] computeLogWeights(List<Region> regions) {
        double[] logWeights = new double[regions.size()];
        for (int i = 0; i < regions.size(); i++) {
            Region region = regions.get(i);
            logWeights[i] = Math.log(region.upperHeight() - region.lowerHeight()) + this.computeLogDensity(region);
        }
        return logWeights;
    }

    /**
     * Computes the log grafting density log g at the midpoint of the given region, i.e.
     * f (-λτ + N log(λτ / 3L)) with τ the distance of the midpoint above X.
     */
    private double computeLogDensity(Region region) {
        int numSites = this.subtreeSequence.length;
        double distance = 0.5 * (region.lowerHeight() + region.upperHeight()) - this.x.getHeight();

        double logDensity = -this.totalRate * distance
                + region.numDifferences() * Math.log(this.totalRate * distance / (3.0 * numSites));
        return this.annealing * logDensity;
    }

    /** Computes the total mutation rate λ(X) of the given sequence on the branch above X. */
    private double computeTotalRate(int[] sequence) {
        int numStates = this.mutations.getAlignment().getMaxStateCount();

        double totalRate = 0.0;
        for (int state : sequence) {
            totalRate -= this.branchRateMatrix[state * numStates + state];
        }
        return totalRate;
    }

    private double computeLogSum(double[] logValues) {
        double max = Double.NEGATIVE_INFINITY;
        for (double logValue : logValues) {
            max = Math.max(max, logValue);
        }

        double sum = 0.0;
        for (double logValue : logValues) {
            sum += Math.exp(logValue - max);
        }
        return max + Math.log(sum);
    }

}
