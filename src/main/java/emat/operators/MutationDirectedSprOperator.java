package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.helper.StochasticMapping;
import emat.state.Mutation;

import java.util.ArrayList;
import java.util.List;

@Description("Regrafts a random subtree preferably where the sequence is close to the sequence of the subtree, " +
        "using the mutations of the EMAT as a parsimony map (mutation-directed SPR, docs/mcmc-moves.md §8.3). " +
        "The root is never changed.")
public class MutationDirectedSprOperator extends SubtreePruneRegraftOperator {

    final public Input<Double> annealingInput = new Input<>("annealing", "the power f to which the grafting density is raised, which makes the proposals less eager", 0.8);

    double annealing;

    /**
     * A part of a branch of the pruned tree with a constant sequence: the node below the
     * branch, the height range of the part that is older than X, and the number of sites
     * where its sequence differs from the sequence of X.
     */
    record Region(Node node, double lowerHeight, double upperHeight, int numDifferences) {
    }

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.annealing = this.annealingInput.get();
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
     * Cuts every branch of the pruned tree at its mutations into regions of constant
     * sequence, picks a region with probability proportional to its length times the
     * grafting density g at its midpoint, and draws the height uniformly within it. Under
     * Jukes–Cantor with rate λ(X) spread over L sites, the density of attaching at distance
     * τ above X to a sequence with N differences is g = [exp(-λτ) (λτ / 3L)^N]^f
     * (Eq. 44). The pruned tree, its mutations and the sequence of X are the same before and
     * after the move, so the total weight cancels and the Hastings ratio is the ratio of g
     * at the midpoints of the old and the new region.
     */
    @Override
    protected GraftingPoint proposeGraftingPoint(Node x) {
        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);

        int[] subtreeSequence = MutationPaths.getSequence(this.mutations, x);
        double totalRate = this.computeTotalRate(subtreeSequence);

        // collect the regions of the pruned tree, starting from the differences at the root

        Node root = this.tree.getRoot();
        int[] rootSequence = MutationPaths.getSequence(this.mutations, root);

        int rootDifferences = 0;
        for (int site = 0; site < subtreeSequence.length; site++) {
            if (rootSequence[site] != subtreeSequence[site]) {
                rootDifferences++;
            }
        }

        List<Region> regions = new ArrayList<>();
        for (Node child : root.getChildren()) {
            this.collectRegions(child, root.getHeight(), rootDifferences, x, subtreeSequence, regions);
        }

        // pick a region by its weight, and a height uniformly within it

        double[] logWeights = new double[regions.size()];
        double maxLogWeight = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < regions.size(); i++) {
            Region region = regions.get(i);
            logWeights[i] = Math.log(region.upperHeight() - region.lowerHeight()) + this.computeLogDensity(region, x, totalRate);
            maxLogWeight = Math.max(maxLogWeight, logWeights[i]);
        }

        double[] weights = new double[regions.size()];
        for (int i = 0; i < regions.size(); i++) {
            weights[i] = Math.exp(logWeights[i] - maxLogWeight);
        }

        Region newRegion = regions.get(StochasticMapping.sampleIndex(weights));
        double newParentHeight = newRegion.lowerHeight() + Randomizer.nextDouble() * (newRegion.upperHeight() - newRegion.lowerHeight());

        // the old attachment point lies on the joined branch above S in the pruned tree

        Region oldRegion = this.findRegion(regions, sibling, parent.getHeight());

        double logHastingsRatio = this.computeLogDensity(oldRegion, x, totalRate) - this.computeLogDensity(newRegion, x, totalRate);
        return new GraftingPoint(newRegion.node(), newParentHeight, logHastingsRatio);
    }

    /**
     * Collects the regions on the branch of the pruned tree above the given node, which
     * starts at the given height with the given number of differences to X, and then on
     * every branch below it. In the pruned tree, the subtree below X is removed and the
     * branches above P and above S are joined.
     */
    private void collectRegions(Node node, double branchStartHeight, int numDifferences, Node x, int[] subtreeSequence, List<Region> regions) {
        Node parent = x.getParent();
        if (node == x) {
            return;
        }
        if (node == parent) {
            // join the branch above P with the branch above S
            Node sibling = this.getOtherChild(parent, x);
            List<Mutation> joinedMutations = new ArrayList<>(this.mutations.getMutations(parent));
            joinedMutations.addAll(this.mutations.getMutations(sibling));
            this.collectBranchRegions(sibling, branchStartHeight, numDifferences, joinedMutations, x, subtreeSequence, regions);
            return;
        }

        this.collectBranchRegions(node, branchStartHeight, numDifferences, this.mutations.getMutations(node), x, subtreeSequence, regions);
    }

    /**
     * Cuts the branch above the given node at the given mutations, which are sorted by
     * descending height, into regions, keeping only the parts older than X. Then continues
     * with the branches below the node.
     */
    private void collectBranchRegions(Node node, double branchStartHeight, int numDifferences, List<Mutation> branchMutations,
                                      Node x, int[] subtreeSequence, List<Region> regions) {
        double upperHeight = branchStartHeight;

        for (Mutation mutation : branchMutations) {
            this.addRegion(node, mutation.time(), upperHeight, numDifferences, x, regions);

            // a mutation changes the number of differences at its site
            int endState = subtreeSequence[mutation.site()];
            numDifferences += (mutation.newState() != endState ? 1 : 0) - (mutation.oldState() != endState ? 1 : 0);
            upperHeight = mutation.time();
        }

        this.addRegion(node, node.getHeight(), upperHeight, numDifferences, x, regions);

        for (Node child : node.getChildren()) {
            this.collectRegions(child, node.getHeight(), numDifferences, x, subtreeSequence, regions);
        }
    }

    /** Adds the part of the given height range that is older than X as a region, if there is one. */
    private void addRegion(Node node, double lowerHeight, double upperHeight, int numDifferences, Node x, List<Region> regions) {
        lowerHeight = Math.max(lowerHeight, x.getHeight());
        if (upperHeight > lowerHeight) {
            regions.add(new Region(node, lowerHeight, upperHeight, numDifferences));
        }
    }

    /** Returns the region above the given node that contains the given height. */
    private Region findRegion(List<Region> regions, Node node, double height) {
        for (Region region : regions) {
            if (region.node() == node && region.lowerHeight() < height && height <= region.upperHeight()) {
                return region;
            }
        }
        throw new IllegalStateException("The old attachment point lies in no region.");
    }

    /**
     * Computes the log grafting density log g at the midpoint of the given region, i.e.
     * f (-λτ + N log(λτ / 3L)) with τ the distance of the midpoint above X.
     */
    private double computeLogDensity(Region region, Node x, double totalRate) {
        int numSites = this.mutations.getAlignment().getSiteCount();
        double distance = 0.5 * (region.lowerHeight() + region.upperHeight()) - x.getHeight();

        double logDensity = -totalRate * distance
                + region.numDifferences() * Math.log(totalRate * distance / (3.0 * numSites));
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

}
