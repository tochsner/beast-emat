package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.helper.FitchParsimony;
import emat.helper.MutationPaths;
import emat.helper.SiteHistorySampler;
import emat.helper.SiteMutations;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.UniformisedStochasticMapping;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

@Description("Resamples the history of a single site on a connected region around an inner node, the centre, " +
        "from its exact conditional distribution given the rest of the history. The region grows breadth-first " +
        "from the centre to at most maxNodes inner nodes, whose states are resampled together with the paths on " +
        "all branches at these nodes. With one node, the region is the star around the centre, and with enough " +
        "nodes the whole history of the site is resampled. The states of the inner nodes at the outer ends of " +
        "the region stay fixed, and tips at its ends keep a state compatible with their data. The centre and the " +
        "site are chosen by the selection strategy. With uniform selection, the move is a Gibbs move that is " +
        "always accepted. With mutation-anchored selection, the choice depends on the mutations, so the move is " +
        "accepted by Metropolis-Hastings.")
public class SiteRegionGibbsOperator extends Operator {

    /** The strategy for choosing the centre of the region and the site. */
    public enum Selection {
        UNIFORM,
        MUTATION_ANCHORED
    }

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Integer> maxNodesInput = new Input<>("maxNodes", "the maximum number of inner nodes whose states are resampled. The region grows breadth-first from the centre until it holds this many inner nodes or the whole tree", 1);
    final public Input<Selection> selectionInput = new Input<>("selection", "the strategy for choosing the centre of the region and the site: UNIFORM picks an inner node (including the root) uniformly and a site uniformly or by parsimony score, MUTATION_ANCHORED picks a random mutation below the root and one of the inner nodes at the ends of its branch, mixed with uniform selection", Selection.UNIFORM, Selection.values());
    final public Input<Double> uniformFractionInput = new Input<>("uniformFraction", "the probability of selecting uniformly instead of anchored at a mutation, which must be positive so that regions without mutations can be reached. Only used by MUTATION_ANCHORED", 0.1);

    final public Input<Boolean> parsimonySitesInput = new Input<>("parsimonySites", "whether a site that is not picked through a mutation is picked proportional to its Fitch parsimony score on the UPGMA tree of the alignment plus the pseudo count instead of uniformly", false);
    final public Input<Double> pseudoCountInput = new Input<>("pseudoCount", "the weight added to the parsimony score of every site, so that sites without parsimony changes are also resampled. Only used with parsimonySites", 1.0);
    final public Input<Integer> maxNumSamplesInput = new Input<>("maxNumSamples", "the maximum number of sequences used to build the UPGMA tree for the parsimony scores. Larger alignments are uniformly subsampled", 1000);

    /** The centre of the region and the site whose history on it is resampled. */
    private record RegionChoice(Node centre, int site) {
    }

    // marks a node of the region whose state is resampled
    private static final int FREE_STATE = -1;

    Mutations mutations;
    TreeInterface tree;
    int maxNodes;
    Selection selection;
    double uniformFraction;
    SiteHistorySampler siteHistorySampler;

    int numSites;

    // reusable per node: the state under the old history of the region
    int[] oldStates;

    // the cumulative weights of the sites for picking a site without a mutation
    double[] cumulativeSiteWeights;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.maxNodes = this.maxNodesInput.get();
        this.selection = this.selectionInput.get();
        this.uniformFraction = this.uniformFractionInput.get();
        this.numSites = this.mutations.getReferenceSequence().length;
        this.oldStates = new int[this.tree.getNodeCount()];

        this.cumulativeSiteWeights = this.computeCumulativeSiteWeights();

        if (this.maxNodes < 1) {
            throw new IllegalArgumentException("The maximum number of nodes must be at least 1.");
        }
        if (this.selection == Selection.MUTATION_ANCHORED && !(this.uniformFraction > 0.0 && this.uniformFraction <= 1.0)) {
            throw new IllegalArgumentException("The uniform fraction must lie in (0, 1].");
        }

        int numStates = this.mutations.getAlignment().getMaxStateCount();
        this.siteHistorySampler = new SiteHistorySampler(this.mutations, this.geneticPriorInput.get(), new UniformisedStochasticMapping(numStates));
    }

    /**
     * Picks the centre of the region and the site by the selection strategy and replaces the
     * history of the site on the region by a draw from its exact conditional distribution.
     * The region only depends on the centre and the tree. With uniform selection, the choice
     * does not depend on the state, so as a Gibbs move it is always accepted. Otherwise, the
     * Hastings ratio is q(choice | new) p(old region) / (q(choice | old) p(new region)),
     * where p is the conditional distribution of the region history. As the posterior ratio
     * equals p(new region) / p(old region), the acceptance probability reduces to the ratio
     * of the selection probabilities.
     */
    @Override
    public double proposal() {
        if (this.tree.getInternalNodeCount() == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        int numMutations = this.selection == Selection.UNIFORM ? 0 : this.mutations.getMutationCount();
        RegionChoice choice = this.selectRegion(numMutations);
        Node centre = choice.centre();
        int site = choice.site();

        // collect the region, its old history and the states that stay fixed

        Set<Node> resampledNodes = this.collectResampledNodes(centre);
        List<Node> regionNodes = this.collectRegionNodes(centre, resampledNodes);

        List<List<Mutation>> oldRegionMutations = new ArrayList<>();
        for (Node node : regionNodes) {
            oldRegionMutations.add(SiteMutations.collect(this.mutations, node, site));
        }

        int startState = this.getStartState(regionNodes.getFirst(), site, oldRegionMutations.getFirst());
        int[] fixedStates = this.collectFixedStates(regionNodes, resampledNodes, oldRegionMutations, startState);

        // resample the region

        this.siteHistorySampler.updateModel(regionNodes);
        List<List<Mutation>> newRegionMutations = this.siteHistorySampler.sampleRegionHistory(site, startState, regionNodes, fixedStates);

        double logHastingsRatio = Double.POSITIVE_INFINITY;
        if (this.selection != Selection.UNIFORM) {
            int numNewMutations = numMutations - this.countNonRootMutations(regionNodes, oldRegionMutations)
                    + this.countNonRootMutations(regionNodes, newRegionMutations);

            logHastingsRatio = this.computeLogSelectionProbability(centre, site, regionNodes, newRegionMutations, numNewMutations)
                    - this.computeLogSelectionProbability(centre, site, regionNodes, oldRegionMutations, numMutations)
                    + this.siteHistorySampler.computeLogRegionHistoryDensity(site, startState, regionNodes, fixedStates, oldRegionMutations)
                    - this.siteHistorySampler.computeLogRegionHistoryDensity(site, startState, regionNodes, fixedStates, newRegionMutations);
        }

        // replace the histories on the region

        for (int i = 0; i < regionNodes.size(); i++) {
            SiteMutations.replace(this.mutations, regionNodes.get(i), site, newRegionMutations.get(i), this);
        }

        return logHastingsRatio;
    }

    /* Region */

    /**
     * Collects the inner nodes whose states are resampled: the centre and the inner nodes
     * closest to it in number of branches, up to the maximum number of nodes. Among the
     * neighbours of a node, its parent comes before its children.
     */
    private Set<Node> collectResampledNodes(Node centre) {
        Set<Node> resampledNodes = new HashSet<>();

        // the queue grows while it is traversed, which yields the breadth-first order

        List<Node> queue = new ArrayList<>();
        Set<Node> queuedNodes = new HashSet<>();
        queue.add(centre);
        queuedNodes.add(centre);

        for (int i = 0; i < queue.size() && resampledNodes.size() < this.maxNodes; i++) {
            Node node = queue.get(i);
            resampledNodes.add(node);

            if (!node.isRoot() && queuedNodes.add(node.getParent())) {
                queue.add(node.getParent());
            }
            for (Node child : node.getChildren()) {
                if (!child.isLeaf() && queuedNodes.add(child)) {
                    queue.add(child);
                }
            }
        }

        return resampledNodes;
    }

    /**
     * Collects the nodes below the branches of the region, i.e. the resampled nodes and
     * their children. The list starts with the top node of the region, and every other node
     * comes after its parent.
     */
    private List<Node> collectRegionNodes(Node centre, Set<Node> resampledNodes) {
        Node top = centre;
        while (!top.isRoot() && resampledNodes.contains(top.getParent())) {
            top = top.getParent();
        }

        List<Node> regionNodes = new ArrayList<>();
        regionNodes.add(top);

        // the list grows while it is traversed, which yields the breadth-first order
        for (int i = 0; i < regionNodes.size(); i++) {
            Node node = regionNodes.get(i);
            if (resampledNodes.contains(node)) {
                regionNodes.addAll(node.getChildren());
            }
        }

        return regionNodes;
    }

    /**
     * Collects the states that stay fixed, indexed like the region nodes: an inner node of
     * the region that is not resampled keeps its state, which follows from the old history.
     * The resampled nodes and the tips are free.
     */
    private int[] collectFixedStates(List<Node> regionNodes, Set<Node> resampledNodes, List<List<Mutation>> oldRegionMutations,
                                     int startState) {
        int[] fixedStates = new int[regionNodes.size()];
        Arrays.fill(fixedStates, FREE_STATE);

        // every node comes after its parent, so the states of the old history follow from the top down

        for (int i = 0; i < regionNodes.size(); i++) {
            Node node = regionNodes.get(i);
            List<Mutation> branchMutations = oldRegionMutations.get(i);

            int branchStartState = i == 0 ? startState : this.oldStates[node.getParent().getNr()];
            this.oldStates[node.getNr()] = branchMutations.isEmpty() ? branchStartState : branchMutations.getLast().newState();

            if (!node.isLeaf() && !resampledNodes.contains(node)) {
                fixedStates[i] = this.oldStates[node.getNr()];
            }
        }

        return fixedStates;
    }

    /**
     * Returns the state of the given site at the top of the branch above the top node of the
     * region, given the mutations of the site on this branch. For the root, this is the
     * reference state.
     */
    private int getStartState(Node top, int site, List<Mutation> topMutations) {
        if (top.isRoot()) {
            return this.mutations.getReferenceSequence()[site];
        }

        // without mutations on the branch, the state at its top equals the state of the top node
        return topMutations.isEmpty() ? MutationPaths.getState(this.mutations, top, site) : topMutations.getFirst().oldState();
    }

    /* Selection */

    /**
     * Picks the centre of the region among the inner nodes and the site. The given number of
     * mutations below the root is only used by mutation-anchored selection, which falls
     * back to uniform selection if there are none.
     */
    private RegionChoice selectRegion(int numMutations) {
        boolean isAnchored = this.selection == Selection.MUTATION_ANCHORED
                && numMutations > 0
                && Randomizer.nextDouble() >= this.uniformFraction;

        return isAnchored ? this.selectAnchoredRegion(numMutations) : this.selectUniformRegion();
    }

    /** Picks an inner node, including the root, uniformly, and a site proportional to its weight. */
    private RegionChoice selectUniformRegion() {
        // BEAST numbers the inner nodes after the leaves
        Node centre = this.tree.getNode(this.tree.getLeafNodeCount() + Randomizer.nextInt(this.tree.getInternalNodeCount()));
        return new RegionChoice(centre, this.selectSite());
    }

    /**
     * Computes the cumulative weights of the sites: all equal, or the Fitch parsimony score
     * of the site on the UPGMA tree of the alignment plus the pseudo count.
     */
    private double[] computeCumulativeSiteWeights() {
        double pseudoCount = this.pseudoCountInput.get();
        if (pseudoCount <= 0.0) {
            throw new IllegalArgumentException("The pseudo count must be positive.");
        }

        int[] siteScores = this.parsimonySitesInput.get()
                ? FitchParsimony.computeOnUpgmaTree(this.mutations.getAlignment(), this.maxNumSamplesInput.get()).getSiteScores()
                : null;

        double[] cumulativeWeights = new double[this.numSites];
        double totalWeight = 0.0;
        for (int site = 0; site < this.numSites; site++) {
            totalWeight += siteScores == null ? 1.0 : siteScores[site] + pseudoCount;
            cumulativeWeights[site] = totalWeight;
        }

        return cumulativeWeights;
    }

    /** Picks a site proportional to its weight. */
    private int selectSite() {
        double threshold = Randomizer.nextDouble() * this.cumulativeSiteWeights[this.numSites - 1];

        // find the first site whose cumulative weight exceeds the threshold

        int low = 0;
        int high = this.numSites - 1;
        while (low < high) {
            int middle = (low + high) >>> 1;
            if (this.cumulativeSiteWeights[middle] > threshold) {
                high = middle;
            } else {
                low = middle + 1;
            }
        }

        return low;
    }

    /**
     * Picks a mutation uniformly among the given number of mutations below the root, and
     * one of the inner nodes at the ends of its branch with equal probability. The lower end
     * of a branch above a tip is not an inner node, so the upper end is always picked there.
     */
    private RegionChoice selectAnchoredRegion(int numMutations) {
        int mutationNr = Randomizer.nextInt(numMutations);

        for (Node node : this.tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }

            List<Mutation> branchMutations = this.mutations.getMutations(node);
            if (mutationNr < branchMutations.size()) {
                Node centre = node.isLeaf() || Randomizer.nextBoolean() ? node.getParent() : node;
                return new RegionChoice(centre, branchMutations.get(mutationNr).site());
            }
            mutationNr -= branchMutations.size();
        }

        throw new IllegalStateException("The number of mutations below the root is out of date.");
    }

    /**
     * Computes the log probability of choosing the given centre and the site of the given
     * region history, if the region carries the given history and the tree carries the given
     * number of mutations below the root. Only the mutations on the branches at the centre
     * can pick it, which all belong to the region: a mutation on the branch above the centre
     * or above an inner child picks the centre with probability 1/2, and a mutation on the
     * branch above a tip child always picks it.
     */
    private double computeLogSelectionProbability(Node centre, int site, List<Node> regionNodes, List<List<Mutation>> regionMutations,
                                                  int numMutations) {
        double siteWeight = this.cumulativeSiteWeights[site] - (site == 0 ? 0.0 : this.cumulativeSiteWeights[site - 1]);
        double uniformProbability = siteWeight / this.cumulativeSiteWeights[this.numSites - 1] / this.tree.getInternalNodeCount();
        if (numMutations == 0) {
            return Math.log(uniformProbability);
        }

        double anchoredWeight = 0.0;
        for (int i = 0; i < regionNodes.size(); i++) {
            Node node = regionNodes.get(i);

            if (node == centre && !node.isRoot()) {
                anchoredWeight += 0.5 * regionMutations.get(i).size();
            } else if (node.getParent() == centre) {
                anchoredWeight += (node.isLeaf() ? 1.0 : 0.5) * regionMutations.get(i).size();
            }
        }

        return Math.log(this.uniformFraction * uniformProbability + (1.0 - this.uniformFraction) * anchoredWeight / numMutations);
    }

    /** Counts the mutations of the given region history that lie below the root. */
    private int countNonRootMutations(List<Node> regionNodes, List<List<Mutation>> regionMutations) {
        int count = 0;
        for (int i = 0; i < regionNodes.size(); i++) {
            if (!regionNodes.get(i).isRoot()) {
                count += regionMutations.get(i).size();
            }
        }
        return count;
    }

}
