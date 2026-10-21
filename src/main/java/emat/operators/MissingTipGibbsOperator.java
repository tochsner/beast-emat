package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.helper.ForwardSiteSimulator;
import emat.helper.MissingSites;
import emat.helper.NodeStateLookup;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.List;

@Description("Resamples the history on the branch above a tip at all sites where the data of the tip is missing, " +
        "i.e. compatible with every state. Given the states at the parent of the tip, the history at such a site " +
        "does not depend on the data, so it is drawn by simulating the substitution process forwards along the " +
        "branch. The tip is picked uniformly among the tips with missing sites, which only depends on the data, so " +
        "the move is a Gibbs move that is always accepted.")
public class MissingTipGibbsOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    Mutations mutations;
    GeneticPrior geneticPrior;
    TreeInterface tree;
    MissingSites missingSites;
    ForwardSiteSimulator forwardSimulator;

    // the numbers of the tips with at least one missing site, which are the candidates of the move
    int[] missingTipNrs;

    // reusable lookup of the states at the parent of the picked tip, which start the resampled histories
    NodeStateLookup parentStates;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();
        this.tree = this.mutations.getTree();
        this.forwardSimulator = new ForwardSiteSimulator(this.geneticPrior, this.mutations.getAlignment().getMaxStateCount());

        int numSites = this.mutations.getReferenceSequence().length;
        this.missingSites = new MissingSites(this.mutations, this.tree);
        this.parentStates = new NodeStateLookup(this.mutations, numSites);

        // the tips keep their numbers and data when the tree changes, but not their node objects
        this.missingTipNrs = this.tree.getExternalNodes().stream()
                .filter(tip -> this.missingSites.getSites(tip).length > 0)
                .mapToInt(Node::getNr)
                .toArray();
    }

    /**
     * Picks a tip with missing sites and replaces its history at these sites by a forward
     * simulation from the states at its parent. The mutations at all other sites stay. As
     * the choice of the tip does not depend on the mutations and the new history is drawn
     * from its exact conditional distribution, the move is always accepted.
     */
    @Override
    public double proposal() {
        if (this.missingTipNrs.length == 0 || this.tree.getInternalNodeCount() == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        Node tip = this.tree.getNode(this.missingTipNrs[Randomizer.nextInt(this.missingTipNrs.length)]);
        int[] sites = this.missingSites.getSites(tip);
        BitSet siteSet = this.missingSites.getSiteSet(tip);

        // keep the mutations at the observed sites and simulate the missing ones

        List<Mutation> oldMutations = this.mutations.getMutations(tip);
        List<Mutation> newMutations = new ArrayList<>();
        boolean hasOldMissingMutations = false;
        for (Mutation mutation : oldMutations) {
            if (siteSet.get(mutation.site())) {
                hasOldMissingMutations = true;
            } else {
                newMutations.add(mutation);
            }
        }

        int numObservedMutations = newMutations.size();
        this.simulateMissingSites(tip, sites, newMutations);

        // a branch without mutations at its missing sites before and after stays untouched
        if (!hasOldMissingMutations && newMutations.size() == numObservedMutations) {
            return Double.POSITIVE_INFINITY;
        }

        newMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        this.mutations.applyMutations(tip, newMutations, this);

        return Double.POSITIVE_INFINITY;
    }

    /**
     * Simulates the substitution process forwards along the branch above the given tip at
     * every given site, starting from the state at the parent, and adds the mutations to the
     * given list.
     */
    private void simulateMissingSites(Node tip, int[] sites, List<Mutation> branchMutations) {
        this.forwardSimulator.updateModel(this.tree.getRoot());
        double rateScale = this.forwardSimulator.getRateScale(tip);

        this.parentStates.resetFor(tip.getParent());
        for (int site : sites) {
            this.forwardSimulator.simulateBranch(tip, site, this.parentStates.applyAsInt(site), rateScale, branchMutations);
        }
    }

}
