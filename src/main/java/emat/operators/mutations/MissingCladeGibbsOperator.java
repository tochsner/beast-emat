package emat.operators.mutations;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

import emat.helper.MissingSites;
import emat.helper.NodeStateLookup;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.ForwardSiteSimulator;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Description("Resamples the history of the clades whose data is missing around a tip. A tip with missing sites is " +
        "picked uniformly, and for each of its missing sites, the clade grows upwards from the tip as long as all " +
        "tips below are missing at the site, for at most maxLevels levels and never up to the root. Given the state " +
        "at the parent of the top of a clade, its history at the site does not depend on the data, so it is drawn by " +
        "simulating the substitution process forwards through the clade. The clades only depend on the tree and the " +
        "data, which the move does not change, so the move is a Gibbs move that is always accepted.")
public class MissingCladeGibbsOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Integer> maxLevelsInput = new Input<>("maxLevels", "the maximum number of levels a clade grows above the picked tip. With 0, only the branch above the tip is resampled", Integer.MAX_VALUE);

    /** The top of a clade and the sites at which its history is resampled. */
    private record Region(Node top, BitSet sites) {
    }

    Mutations mutations;
    TreeInterface tree;
    MissingSites missingSites;
    ForwardSiteSimulator forwardSimulator;
    int maxLevels;

    // the numbers of the tips with at least one missing site, which are the candidates of the move
    int[] missingTipNrs;

    // reusable lookup of the states at the parent of the top of a clade, which start its resampled histories
    NodeStateLookup parentStates;

    // reusable per node: the lowest region of the current proposal that covers the branch above the node, valid if
    // stamped with the current proposal
    int[] lowestRegions;
    long[] lowestRegionProposals;
    long proposalCount;

    // reusable per node: the state at the node after a simulated mutation on its branch, valid if
    // stamped with the epoch of the current site
    int[] endStates;
    long[] endStateEpochs;
    long epoch;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.maxLevels = this.maxLevelsInput.get();
        this.forwardSimulator = new ForwardSiteSimulator(this.geneticPriorInput.get(), this.mutations.getAlignment().getMaxStateCount());

        if (this.maxLevels < 0) {
            throw new IllegalArgumentException("The maximum number of levels must not be negative.");
        }

        int numSites = this.mutations.getReferenceSequence().length;
        this.missingSites = new MissingSites(this.mutations, this.tree);
        this.parentStates = new NodeStateLookup(this.mutations, numSites);
        this.lowestRegions = new int[this.tree.getNodeCount()];
        this.lowestRegionProposals = new long[this.tree.getNodeCount()];
        this.endStates = new int[this.tree.getNodeCount()];
        this.endStateEpochs = new long[this.tree.getNodeCount()];

        // the tips keep their numbers and data when the tree changes, but not their node objects
        this.missingTipNrs = this.tree.getExternalNodes().stream()
                .filter(tip -> this.missingSites.getSites(tip).length > 0)
                .mapToInt(Node::getNr)
                .toArray();
    }

    /**
     * Picks a tip with missing sites, collects the missing clades around it and replaces
     * their histories by forward simulations from the states at their parents. The
     * mutations at all other sites and branches stay. The new histories are simulated on
     * the old ones and only applied at the end, so that every start state is read from the
     * history before the move. The regions are nested, as their tops lie on the path from
     * the tip upwards, so a branch is resampled at the sites of the lowest region that
     * covers it and of all regions above.
     */
    @Override
    public double proposal() {
        if (this.missingTipNrs.length == 0 || this.tree.getInternalNodeCount() == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        Node tip = this.tree.getNode(this.missingTipNrs[Randomizer.nextInt(this.missingTipNrs.length)]);
        List<Region> regions = this.collectRegions(tip);

        // simulate every region, collecting the new mutations and the covered branches

        Map<Integer, List<Mutation>> newMutations = new HashMap<>();
        List<Node> coveredNodes = new ArrayList<>();

        this.proposalCount++;
        this.forwardSimulator.updateModel(this.tree.getRoot());
        for (int index = 0; index < regions.size(); index++) {
            this.simulateRegion(regions.get(index), index, newMutations, coveredNodes);
        }

        this.applyRegions(regions, newMutations, coveredNodes);

        return Double.POSITIVE_INFINITY;
    }

    /* Regions */

    /**
     * Collects the clades around the given tip: each missing site of the tip stops at the
     * highest ancestor below which all tips are missing at the site, within the maximum
     * number of levels and below the root. Sites that stop at the same node form one region.
     */
    private List<Region> collectRegions(Node tip) {
        List<Region> regions = new ArrayList<>();
        BitSet remainingSites = (BitSet) this.missingSites.getSiteSet(tip).clone();

        Node current = tip;
        for (int level = 0; level < this.maxLevels && !remainingSites.isEmpty() && !current.getParent().isRoot(); level++) {
            Node parent = current.getParent();

            // a site continues upwards if all tips below the siblings are missing there as well

            BitSet continuingSites = (BitSet) remainingSites.clone();
            for (Node sibling : parent.getChildren()) {
                if (sibling != current) {
                    this.intersectMissingSites(sibling, continuingSites);
                }
            }

            BitSet stoppingSites = (BitSet) remainingSites.clone();
            stoppingSites.andNot(continuingSites);
            if (!stoppingSites.isEmpty()) {
                regions.add(new Region(current, stoppingSites));
            }

            remainingSites = continuingSites;
            current = parent;
        }

        if (!remainingSites.isEmpty()) {
            regions.add(new Region(current, remainingSites));
        }

        return regions;
    }

    /**
     * Removes from the given sites those where any tip below the given node is not
     * missing, and stops as soon as no site is left.
     */
    private void intersectMissingSites(Node node, BitSet sites) {
        Deque<Node> stack = new ArrayDeque<>();
        stack.push(node);

        while (!stack.isEmpty() && !sites.isEmpty()) {
            Node next = stack.pop();
            if (next.isLeaf()) {
                sites.and(this.missingSites.getSiteSet(next));
            } else {
                for (Node child : next.getChildren()) {
                    stack.push(child);
                }
            }
        }
    }

    /* Simulation */

    /**
     * Simulates the history of every site of the given region through the clade below its
     * top, starting from the state at the parent of the top, and records the new mutations
     * and the branches of the clade that no lower region covers. The branches are laid end to end
     * in expected substitutions, so that each site needs a single stream of events of the
     * uniformised process for the whole clade, instead of a draw per branch.
     */
    private void simulateRegion(Region region, int index, Map<Integer, List<Mutation>> newMutations, List<Node> coveredNodes) {
        Node top = region.top();
        List<Node> cladeNodes = this.collectPreOrder(top);

        double[] rateScales = new double[cladeNodes.size()];
        double[] branchStarts = new double[cladeNodes.size() + 1];
        for (int i = 0; i < cladeNodes.size(); i++) {
            Node node = cladeNodes.get(i);
            rateScales[i] = this.forwardSimulator.getRateScale(node);
            branchStarts[i + 1] = branchStarts[i] + node.getLength() * rateScales[i];

            // the regions come from the bottom up, so the first region to reach a branch is its lowest
            if (this.lowestRegionProposals[node.getNr()] != this.proposalCount) {
                this.lowestRegionProposals[node.getNr()] = this.proposalCount;
                this.lowestRegions[node.getNr()] = index;
                coveredNodes.add(node);
            }
        }

        this.parentStates.resetFor(top.getParent());
        for (int site = region.sites().nextSetBit(0); site >= 0; site = region.sites().nextSetBit(site + 1)) {
            this.simulateSite(top, cladeNodes, rateScales, branchStarts, newMutations, site, this.parentStates.applyAsInt(site));
        }
    }

    /**
     * Simulates a single site through the clade: the events of the uniformised process are
     * spread over the branches laid end to end, and as the branches come in pre-order, each
     * branch is reached after all branches above it. Each event is resolved in the state at
     * its position, which is the end state of the closest branch above with a mutation, or
     * the start state of the clade.
     */
    private void simulateSite(Node top, List<Node> cladeNodes, double[] rateScales, double[] branchStarts,
                              Map<Integer, List<Mutation>> newMutations, int site, int topState) {
        this.epoch++;
        double eventRate = this.forwardSimulator.getMaxEscapeRate();
        double totalLength = branchStarts[cladeNodes.size()];

        int branch = -1;
        int state = topState;
        double previousHeight = 0.0;
        for (double position = Randomizer.nextExponential(eventRate); position < totalLength; position += Randomizer.nextExponential(eventRate)) {
            int index = this.findBranch(branchStarts, position);
            Node node = cladeNodes.get(index);
            if (index != branch) {
                branch = index;
                state = this.getStartState(node, top, topState);
                previousHeight = node.getParent().getHeight();
            }

            int newState = this.forwardSimulator.sampleUniformisedJump(state);
            if (newState == state) {
                continue;
            }

            double height = node.getParent().getHeight() - (position - branchStarts[index]) / rateScales[index];
            newMutations.computeIfAbsent(node.getNr(), nr -> new ArrayList<>()).add(new Mutation(node.getNr(), height, previousHeight, site, state, newState));

            previousHeight = height;
            state = newState;
            this.endStates[node.getNr()] = newState;
            this.endStateEpochs[node.getNr()] = this.epoch;
        }
    }

    /** Returns the index of the branch whose stretch of the laid out branches contains the given position. */
    private int findBranch(double[] branchStarts, double position) {
        int low = 0;
        int high = branchStarts.length - 2;
        while (low < high) {
            int middle = (low + high + 1) >>> 1;
            if (branchStarts[middle] <= position) {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        return low;
    }

    /**
     * Returns the state at the top of the branch above the given node: the end state of the
     * closest ancestor within the clade whose branch got a mutation at the current site, or
     * the start state of the clade.
     */
    private int getStartState(Node node, Node top, int topState) {
        for (Node ancestor = node; ancestor != top; ) {
            ancestor = ancestor.getParent();
            if (this.endStateEpochs[ancestor.getNr()] == this.epoch) {
                return this.endStates[ancestor.getNr()];
            }
        }
        return topState;
    }

    /** Collects the given node and all nodes below it, with every node after its parent. */
    private List<Node> collectPreOrder(Node top) {
        List<Node> nodes = new ArrayList<>();
        nodes.add(top);

        // the list grows while it is traversed, which yields the breadth-first order
        for (int i = 0; i < nodes.size(); i++) {
            nodes.addAll(nodes.get(i).getChildren());
        }

        return nodes;
    }

    /**
     * Replaces the mutations at the resampled sites of every covered branch by the simulated
     * ones: the sites of its lowest region and of all regions above. Branches without
     * mutations at their resampled sites before and after stay untouched.
     */
    private void applyRegions(List<Region> regions, Map<Integer, List<Mutation>> newMutations, List<Node> coveredNodes) {

        // the sites resampled on a branch whose lowest region has the given index

        BitSet[] resampledSites = new BitSet[regions.size()];
        BitSet sitesAbove = new BitSet();
        for (int index = regions.size() - 1; index >= 0; index--) {
            sitesAbove.or(regions.get(index).sites());
            resampledSites[index] = (BitSet) sitesAbove.clone();
        }

        for (Node node : coveredNodes) {
            BitSet sites = resampledSites[this.lowestRegions[node.getNr()]];
            List<Mutation> simulatedMutations = newMutations.getOrDefault(node.getNr(), List.of());

            List<Mutation> branchMutations = new ArrayList<>();
            boolean hasOldResampledMutations = false;
            for (Mutation mutation : this.mutations.getMutations(node)) {
                if (sites.get(mutation.site())) {
                    hasOldResampledMutations = true;
                } else {
                    branchMutations.add(mutation);
                }
            }

            if (!hasOldResampledMutations && simulatedMutations.isEmpty()) {
                continue;
            }

            branchMutations.addAll(simulatedMutations);
            branchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
            this.mutations.applyMutations(node, branchMutations, this);
        }
    }

}
