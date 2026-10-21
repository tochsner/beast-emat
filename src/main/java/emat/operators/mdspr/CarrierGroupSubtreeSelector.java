package emat.operators.mdspr;

import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Picks the carrier groups of homoplastic sites that switch between sitting apart and sitting
 * with the other carriers of their state, mixed with a share of picks in proportion to the
 * branch length. A carrier group is the set of tips below a branch whose last mutation at a
 * site leads to a state that another such branch at the same site leads to as well. It sits
 * apart in a sample if it is such a group.
 * <p>
 * The groups are learned from the history of the chain: every given number of proposals, the
 * carrier groups of all sites are sampled. After a burn-in, the samples of a given length are
 * used to estimate, for every group, how often it sits apart and the autocorrelation time of
 * that indicator. Then the weights are frozen. A group that sits apart in at least the minimum
 * and at most one minus the minimum share of the samples gets the estimated time raised to
 * the power as its weight, and every other group none. Before the weights are frozen, and
 * whenever no group can be picked, all picks are in proportion to the branch length, or there
 * is no pick at all without a floor.
 * <p>
 * In a state, a group can be picked through the node whose clade it is, if it is a clade. The
 * move keeps the picked subtree a clade, so a group picked to join the other carriers of its
 * state can be picked again to leave them.
 */
public class CarrierGroupSubtreeSelector extends SubtreeSelector {

    private final Mutations mutations;
    private final double power;
    private final double floor;
    private final double minFrequency;
    private final int interval;
    private final int burnin;
    private final int numLearningSamples;

    // the number of calls so far that count towards the sampling interval, and the number of samples so far
    private long numCalls;
    private long numSamples;

    // per group, as the set of its tip numbers, how often it sat apart, how often that changed between two samples, and whether it sat apart in the last sample
    private final Map<BitSet, GroupStatistics> groupStatistics = new HashMap<>();
    private Set<BitSet> lastApartGroups = new HashSet<>();
    private int numEstimateSamples;

    // after the learning, the tips and the weight of every group with a positive weight
    private boolean isFrozen;
    private final List<int[]> frozenGroupTips = new ArrayList<>();
    private final List<Double> frozenGroupWeights = new ArrayList<>();

    // in the current state, the share of the carrier groups per node and the total branch length of the candidates
    private final double[] groupNodeWeights;
    private double totalBranchLength;
    private boolean hasGroupCandidates;

    // the buffers of the most recent common ancestor search
    private final int[] pathStamps;
    private final int[] pathPositions;
    private int pathStamp;

    /**
     * @param power        the power to which the estimated autocorrelation time of a group is raised. With 0, all switching groups are picked equally often
     * @param floor        the share of the picks in proportion to the branch length, so that every subtree keeps being picked
     * @param minFrequency the smallest share of the samples in which a group must sit apart, and with the other carriers, to be picked
     * @param interval     the number of proposals between two samples of the carrier groups
     * @param burnin       the number of samples that are discarded before the learning starts
     * @param numLearningSamples the number of samples that the weights are estimated from before they are frozen
     */
    public CarrierGroupSubtreeSelector(Tree tree, Mutations mutations, double power, double floor, double minFrequency, int interval, int burnin, int numLearningSamples) {
        super(tree);
        if (power < 0.0) {
            throw new IllegalArgumentException("carrierGroupPower must not be negative.");
        }
        if (floor < 0.0 || floor > 1.0) {
            throw new IllegalArgumentException("carrierGroupFloor must be between 0 and 1.");
        }
        if (minFrequency < 0.0 || minFrequency >= 0.5) {
            throw new IllegalArgumentException("carrierGroupMinFrequency must be at least 0 and less than 0.5.");
        }
        if (interval < 1) {
            throw new IllegalArgumentException("carrierGroupInterval must be positive.");
        }
        if (burnin < 0) {
            throw new IllegalArgumentException("carrierGroupBurnin must not be negative.");
        }
        if (numLearningSamples < 3) {
            throw new IllegalArgumentException("carrierGroupLearningSamples must be at least 3.");
        }

        this.mutations = mutations;
        this.power = power;
        this.floor = floor;
        this.minFrequency = minFrequency;
        this.interval = interval;
        this.burnin = burnin;
        this.numLearningSamples = numLearningSamples;

        int numNodes = tree.getNodeCount();
        this.groupNodeWeights = new double[numNodes];
        this.pathStamps = new int[numNodes];
        this.pathPositions = new int[numNodes];
    }

    /* Learning */

    @Override
    public void prepare() {
        if (this.isFrozen) {
            return;
        }

        boolean isSampling = this.numCalls % this.interval == 0;
        this.numCalls++;
        if (!isSampling) {
            return;
        }

        this.numSamples++;
        if (this.numSamples <= this.burnin) {
            return;
        }

        this.sampleCarrierGroups();
        if (this.numEstimateSamples >= this.numLearningSamples) {
            this.freezeWeights();
        }
    }

    /** Records which groups sit apart in the current state, and which of them changed since the previous sample. */
    private void sampleCarrierGroups() {
        Set<BitSet> apartGroups = this.collectApartGroups();

        for (BitSet group : apartGroups) {
            GroupStatistics statistics = this.groupStatistics.computeIfAbsent(group, key -> new GroupStatistics());
            statistics.numApart++;
            if (this.numEstimateSamples > 0 && !this.lastApartGroups.contains(group)) {
                statistics.numSwitches++;
            }
        }
        for (BitSet group : this.lastApartGroups) {
            if (!apartGroups.contains(group)) {
                this.groupStatistics.get(group).numSwitches++;
            }
        }

        this.lastApartGroups = apartGroups;
        this.numEstimateSamples++;
    }

    /**
     * Returns the tip sets below all branches whose last mutation at a site leads to a state
     * that the last mutation of another branch at the same site leads to as well. The branch
     * above the root is left out.
     */
    private Set<BitSet> collectApartGroups() {
        int numStates = this.mutations.getAlignment().getMaxStateCount();
        Map<Long, List<Node>> carrierRoots = new HashMap<>();
        Map<Integer, Mutation> lastMutations = new HashMap<>();

        for (Node node : this.tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }

            // the mutation closest to the node decides the state that the subtree inherits
            lastMutations.clear();
            for (Mutation mutation : this.mutations.getMutations(node)) {
                Mutation last = lastMutations.get(mutation.site());
                if (last == null || mutation.time() < last.time()) {
                    lastMutations.put(mutation.site(), mutation);
                }
            }
            for (Mutation mutation : lastMutations.values()) {
                long key = (long) mutation.site() * numStates + mutation.newState();
                carrierRoots.computeIfAbsent(key, k -> new ArrayList<>(2)).add(node);
            }
        }

        Set<BitSet> apartGroups = new HashSet<>();
        for (List<Node> roots : carrierRoots.values()) {
            if (roots.size() < 2) {
                continue;
            }
            for (Node root : roots) {
                apartGroups.add(this.collectTips(root));
            }
        }
        return apartGroups;
    }

    private BitSet collectTips(Node root) {
        BitSet tips = new BitSet(this.tree.getLeafNodeCount());
        List<Node> stack = new ArrayList<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            Node node = stack.removeLast();
            if (node.isLeaf()) {
                tips.set(node.getNr());
            } else {
                stack.addAll(node.getChildrenMutable());
            }
        }
        return tips;
    }

    /**
     * Estimates, for every group, the share p of the samples in which it sat apart and the
     * autocorrelation time of that indicator as 2 var / E[step²] = 2 p (1 - p) / (switches /
     * (n - 1)), limited to between 1 and the number of samples n, and keeps the groups with
     * p within [m, 1 - m] and at least one switch, weighted by the time raised to the power.
     */
    private void freezeWeights() {
        int n = this.numEstimateSamples;
        for (Map.Entry<BitSet, GroupStatistics> entry : this.groupStatistics.entrySet()) {
            GroupStatistics statistics = entry.getValue();
            double share = (double) statistics.numApart / n;
            if (statistics.numSwitches == 0 || share < this.minFrequency || share > 1.0 - this.minFrequency) {
                continue;
            }

            double act = 2.0 * share * (1.0 - share) * (n - 1) / statistics.numSwitches;
            this.frozenGroupTips.add(entry.getKey().stream().toArray());
            this.frozenGroupWeights.add(Math.pow(Math.min(Math.max(act, 1.0), n), this.power));
        }

        this.isFrozen = true;
        this.groupStatistics.clear();
        this.lastApartGroups.clear();
    }

    /* Weights */

    /**
     * Finds the frozen groups that are clades in the current state, splits the carrier share of
     * the weight among them in proportion to their weights, and sums the branch lengths of the
     * candidates for the floor. Without a group that can be picked, all picks are in
     * proportion to the branch length.
     */
    @Override
    protected void updateWeights() {
        Arrays.fill(this.groupNodeWeights, 0.0);
        this.totalBranchLength = 0.0;
        for (Node node : this.tree.getNodesAsArray()) {
            if (this.isCandidate(node)) {
                this.totalBranchLength += node.getLength();
            }
        }

        double totalGroupWeight = 0.0;
        if (!this.frozenGroupTips.isEmpty()) {
            int[] leafCounts = this.computeLeafCounts();
            for (int i = 0; i < this.frozenGroupTips.size(); i++) {
                int[] tips = this.frozenGroupTips.get(i);
                Node mrca = this.findMostRecentCommonAncestor(tips);
                if (leafCounts[mrca.getNr()] == tips.length && this.isCandidate(mrca)) {
                    this.groupNodeWeights[mrca.getNr()] += this.frozenGroupWeights.get(i);
                    totalGroupWeight += this.frozenGroupWeights.get(i);
                }
            }
        }

        this.hasGroupCandidates = totalGroupWeight > 0.0;
        if (this.hasGroupCandidates) {
            for (int nr = 0; nr < this.groupNodeWeights.length; nr++) {
                this.groupNodeWeights[nr] /= totalGroupWeight;
            }
        }
    }

    private boolean isCandidate(Node node) {
        return !node.isRoot() && !node.getParent().isRoot();
    }

    /** Walks up from the first tip and marks its ancestors, then walks up from every other tip until it meets them. The meeting point closest to the root is the ancestor. */
    private Node findMostRecentCommonAncestor(int[] tips) {
        this.pathStamp++;
        int position = 0;
        for (Node node = this.tree.getNode(tips[0]); node != null; node = node.getParent()) {
            this.pathStamps[node.getNr()] = this.pathStamp;
            this.pathPositions[node.getNr()] = position++;
        }

        Node mrca = this.tree.getNode(tips[0]);
        for (int i = 1; i < tips.length; i++) {
            Node node = this.tree.getNode(tips[i]);
            while (this.pathStamps[node.getNr()] != this.pathStamp) {
                node = node.getParent();
            }
            if (this.pathPositions[node.getNr()] > this.pathPositions[mrca.getNr()]) {
                mrca = node;
            }
        }
        return mrca;
    }

    /** Without a floor, nothing can be picked unless a group can, and the proposal is rejected. */
    @Override
    protected double computeWeight(Node node) {
        double branchLengthWeight = this.totalBranchLength > 0.0 ? node.getLength() / this.totalBranchLength : 0.0;
        if (!this.hasGroupCandidates) {
            return this.floor > 0.0 ? branchLengthWeight : 0.0;
        }
        return this.floor * branchLengthWeight + (1.0 - this.floor) * this.groupNodeWeights[node.getNr()];
    }

    /* Diagnostics */

    @Override
    public int getNumSamples() {
        return this.isFrozen ? this.numLearningSamples : this.numEstimateSamples;
    }

    /** Adds the number of groups with a positive weight to the diagnostics of the picks. */
    @Override
    public void addDiagnostics(Map<String, Double> diagnostics) {
        super.addDiagnostics(diagnostics);
        diagnostics.put("mdspr_carrier_groups", (double) this.frozenGroupTips.size());
    }

    /** The samples of one carrier group: how often it sat apart, and how often that changed between two consecutive samples. */
    private static final class GroupStatistics {
        int numApart;
        int numSwitches;
    }
}
