package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.stochasticmapping.JukesCantorStochasticMapping;
import emat.helper.MutationPaths;
import emat.helper.NodeStateLookup;
import emat.helper.SiteChanges;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

@Description("Reforms the history of a random branch without changing the tree (branch reform, " +
        "docs/mcmc-moves.md §9.2). On a branch below an internal node, all mutation times are redrawn " +
        "uniformly on the branch, keeping the order of the mutations at each site. On a branch below the " +
        "root, the history of the whole path through the root to the other child is resampled by approximate " +
        "Jukes-Cantor stochastic mapping, which lets mutations move between the two root branches and changes " +
        "the root sequence. Missations are not supported.")
public class BranchReformOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    Mutations mutations;
    GeneticPrior geneticPrior;
    TreeInterface tree;
    JukesCantorStochasticMapping stochasticMapping;

    // reusable site maps for the changes from the root down to both root children and the resulting differing sites
    SiteChanges subtreeChanges;
    SiteChanges siblingChanges;
    SiteChanges differingSites;

    // reusable lookup of the states at X for the sites whose end states agree
    NodeStateLookup subtreeStates;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();
        this.tree = this.mutations.getTree();

        int numSites = this.mutations.getReferenceSequence().length;
        this.stochasticMapping = new JukesCantorStochasticMapping(this.mutations.getAlignment().getMaxStateCount(), numSites);

        this.subtreeChanges = new SiteChanges(numSites);
        this.siblingChanges = new SiteChanges(numSites);
        this.differingSites = new SiteChanges(numSites);
        this.subtreeStates = new NodeStateLookup(this.mutations, numSites);
    }

    /**
     * Picks a non-root branch uniformly and reforms its history. The set of non-root
     * branches does not change under the move, so the choice is symmetric.
     */
    @Override
    public double proposal() {
        Node root = this.tree.getRoot();
        int numNodes = this.tree.getNodeCount();
        if (numNodes < 2) {
            return Double.NEGATIVE_INFINITY;
        }

        // sample non-root node
        int nodeNr = Randomizer.nextInt(numNodes);
        Node node = this.tree.getNode(nodeNr);
        while (node == root) {
            nodeNr = Randomizer.nextInt(numNodes);
            node = this.tree.getNode(nodeNr);
        }

        if (node.getParent().isRoot()) {
            return this.reformRootBranches(node);
        } else {
            return this.reformBranch(node);
        }
    }

    /* Branches Below an Internal Node */

    /**
     * Redraws the times of all mutations on the branch above the given node uniformly on
     * the branch. The times drawn for the mutations at one site are sorted and assigned in
     * the original order, so the history at every site keeps its states. For a site with k
     * mutations, the sorted times have density k! / τ^k on a branch of length τ, which does
     * not depend on the old times, so the proposal is symmetric.
     */
    private double reformBranch(Node node) {
        List<Mutation> branchMutations = this.mutations.getMutations(node);
        if (branchMutations.isEmpty()) {
            return Double.NEGATIVE_INFINITY;
        }

        // evolution runs forwards in time from the parent to the node

        double branchStartHeight = node.getParent().getHeight();
        double branchEndHeight = node.getHeight();
        double branchLength = branchStartHeight - branchEndHeight;

        // draw a new time per mutation, grouped by site in the order of the mutations at that site

        Map<Integer, List<Integer>> mutationNrsOfSite = new HashMap<>();
        for (int i = 0; i < branchMutations.size(); i++) {
            mutationNrsOfSite.computeIfAbsent(branchMutations.get(i).site(), site -> new ArrayList<>()).add(i);
        }

        double[] newTimes = new double[branchMutations.size()];

        for (List<Integer> mutationNrs : mutationNrsOfSite.values()) {
            List<Double> siteTimes = new ArrayList<>(mutationNrs.size());
            for (int i = 0; i < mutationNrs.size(); i++) {
                siteTimes.add(branchEndHeight + Randomizer.nextDouble() * branchLength);
            }

            // the mutations are sorted by descending height, so the earliest mutation at a site gets the highest time
            siteTimes.sort(Comparator.reverseOrder());
            for (int i = 0; i < mutationNrs.size(); i++) {
                newTimes[mutationNrs.get(i)] = siteTimes.get(i);
            }
        }

        List<Mutation> newBranchMutations = new ArrayList<>(branchMutations.size());
        for (int i = 0; i < branchMutations.size(); i++) {
            Mutation mutation = branchMutations.get(i);
            newBranchMutations.add(new Mutation(
                    mutation.nodeNr(), newTimes[i], branchStartHeight,
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
        }

        this.mutations.applyMutations(node, this.linkMutations(newBranchMutations, branchStartHeight), this);

        return 0.0;
    }

    /* Branches Below the Root */

    /**
     * Resamples the history on the path from the other root child S up through the root R
     * and down to the given root child X (docs/mcmc-moves.md §5 with P' = P). The sequences
     * of X and S stay fixed. Jukes-Cantor is time-reversible, so the path is mapped as one
     * branch of length τ' = (t_R - t_S) + (t_R - t_X) that starts in the sequence of S and
     * ends in the sequence of X. Its part above R is mirrored onto the R–S branch, with the
     * direction of every mutation reversed, and the state at R becomes the new root state.
     * The real model, including the change of the root prior, enters through the genetic
     * prior. The fictitious Jukes-Cantor rate μ̃ = λ(X) / L only depends on X, which keeps its
     * sequence and branch rate, so the reverse move uses the same rate.
     */
    private double reformRootBranches(Node x) {
        Node root = x.getParent();
        if (root.getChildCount() != 2) {
            return Double.NEGATIVE_INFINITY;
        }

        Node sibling = root.getLeft() == x ? root.getRight() : root.getLeft();

        double rootHeight = root.getHeight();
        double pathStartHeight = 2.0 * rootHeight - sibling.getHeight();
        double jukesCantorRate = this.computeJukesCantorRate(x);

        List<Mutation> oldSubtreeMutations = this.mutations.getMutations(x);
        List<Mutation> oldSiblingMutations = this.mutations.getMutations(sibling);

        // find the sites that differ between S and X, which are the start and end of the path

        MutationPaths.collectChanges(this.mutations, root, sibling, sibling.getHeight(), this.siblingChanges);
        MutationPaths.collectChanges(this.mutations, root, x, x.getHeight(), this.subtreeChanges);
        MutationPaths.combineChanges(this.siblingChanges, this.subtreeChanges, this.differingSites);

        // sample the history on the path while the tree still holds the sequence of X, which the move keeps

        List<Mutation> pathMutations = this.stochasticMapping.sampleBranchHistory(
                x.getNr(), pathStartHeight, x.getHeight(), jukesCantorRate, this.differingSites,
                this.subtreeStates.resetFor(x)
        );

        // the root state at a site is the state of S, unless the mirrored part of the path changes it

        Map<Integer, Integer> newRootStates = new HashMap<>();
        for (Mutation mutation : oldSiblingMutations) {
            newRootStates.put(mutation.site(), mutation.newState());
        }

        // split the path at the root, mirroring the part above it onto the R–S branch

        List<Mutation> newSubtreeMutations = new ArrayList<>();
        List<Mutation> newSiblingMutations = new ArrayList<>();

        for (Mutation mutation : pathMutations) {
            if (mutation.time() < rootHeight) {
                newSubtreeMutations.add(mutation);
                continue;
            }

            // the path runs through its mutations from S up to R, so the last one at a site sets the root state
            newRootStates.put(mutation.site(), mutation.newState());

            double time = Math.min(Math.max(2.0 * rootHeight - mutation.time(), sibling.getHeight()), rootHeight);
            newSiblingMutations.add(new Mutation(
                    sibling.getNr(), time, rootHeight,
                    mutation.site(), mutation.newState(), mutation.oldState()
            ));
        }


        this.mutations.applyMutations(x, this.linkMutations(newSubtreeMutations, rootHeight), this);
        this.mutations.applyMutations(sibling, this.linkMutations(newSiblingMutations, rootHeight), this);
        this.updateRootSequence(root, newRootStates);

        // the path has the same length and differing sites in both directions

        List<Mutation> oldPathMutations = new ArrayList<>(oldSiblingMutations);
        oldPathMutations.addAll(oldSubtreeMutations);

        double logForwardDensity = this.stochasticMapping.computeLogBranchHistoryDensity(
                pathMutations, pathStartHeight, x.getHeight(), jukesCantorRate, this.differingSites.getSize()
        );
        double logBackwardDensity = this.stochasticMapping.computeLogBranchHistoryDensity(
                oldPathMutations, pathStartHeight, x.getHeight(), jukesCantorRate, this.differingSites.getSize()
        );

        return logBackwardDensity - logForwardDensity;
    }

    /**
     * Returns the fictitious Jukes-Cantor rate μ̃ = λ(X) / L of the branch above the given
     * node, with λ(X) the total mutation rate at X, which the genetic prior has cached for
     * the current state, scaled by the branch rate.
     */
    private double computeJukesCantorRate(Node node) {
        double branchRate = this.geneticPrior.branchRateModel.getRateForBranch(node);
        int numSites = this.mutations.getReferenceSequence().length;
        return branchRate * this.geneticPrior.getTotalMutationRate(node) / numSites;
    }

    /**
     * Sets the root state of the given sites. The mutations above the root encode the
     * difference between the reference and the root sequence, so they are rebuilt with a
     * single mutation per differing site. They are left untouched if no root state changes.
     */
    private void updateRootSequence(Node root, Map<Integer, Integer> newRootStates) {
        int[] referenceSequence = this.mutations.getReferenceSequence();

        Map<Integer, Integer> rootStates = new TreeMap<>();
        for (Mutation mutation : this.mutations.getMutations(root)) {
            rootStates.put(mutation.site(), mutation.newState());
        }

        boolean isChanged = false;
        for (Map.Entry<Integer, Integer> entry : newRootStates.entrySet()) {
            int site = entry.getKey();
            int oldState = rootStates.getOrDefault(site, referenceSequence[site]);
            if (oldState != entry.getValue()) {
                rootStates.put(site, entry.getValue());
                isChanged = true;
            }
        }

        if (!isChanged) {
            return;
        }

        // the times of the mutations above the root are meaningless, so they are set to the root height

        double rootHeight = root.getHeight();
        List<Mutation> rootMutations = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : rootStates.entrySet()) {
            int site = entry.getKey();
            if (entry.getValue() != referenceSequence[site]) {
                rootMutations.add(new Mutation(root.getNr(), rootHeight, rootHeight, site, referenceSequence[site], entry.getValue()));
            }
        }

        this.mutations.applyMutations(root, rootMutations, this);
    }

    /* Helpers */

    /**
     * Returns a copy of the given mutations of one branch, sorted by descending height,
     * where every mutation follows the previous mutation at its site, or the branch start
     * if there is none.
     */
    private List<Mutation> linkMutations(List<Mutation> branchMutations, double branchStartHeight) {
        List<Mutation> sortedMutations = new ArrayList<>(branchMutations);
        sortedMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());

        Map<Integer, Double> lastTimeOfSite = new LinkedHashMap<>();
        List<Mutation> linkedMutations = new ArrayList<>(sortedMutations.size());

        for (Mutation mutation : sortedMutations) {
            double timeOfPreviousMutation = lastTimeOfSite.getOrDefault(mutation.site(), branchStartHeight);
            lastTimeOfSite.put(mutation.site(), mutation.time());

            linkedMutations.add(new Mutation(
                    mutation.nodeNr(), mutation.time(), timeOfPreviousMutation,
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
        }

        return linkedMutations;
    }

}
