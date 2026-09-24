package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.helper.SiteHistorySampler;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.prior.GeneticPrior;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Description("Resamples the complete mutational history of a random site from its exact conditional " +
        "distribution (stochastic mapping). This is a Gibbs move, so it is always accepted.")
public class SiteHistoryGibbsOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    Mutations mutations;
    TreeInterface tree;
    SiteHistorySampler siteHistorySampler;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.siteHistorySampler = new SiteHistorySampler(this.mutations, this.geneticPriorInput.get());
    }

    /**
     * Picks a site uniformly and replaces its history by a draw from the conditional
     * distribution of the genetic prior given the tip data and everything else. As a Gibbs
     * move, it is always accepted.
     */
    @Override
    public double proposal() {
        int site = Randomizer.nextInt(this.mutations.getAlignment().getSiteCount());

        this.siteHistorySampler.updateModel();
        List<List<Mutation>> newSiteMutations = this.siteHistorySampler.sampleSiteHistory(site);

        for (Node node : this.tree.getNodesAsArray()) {
            this.replaceSiteMutations(node, site, newSiteMutations.get(node.getNr()));
        }

        return Double.POSITIVE_INFINITY;
    }

    /**
     * Replaces the mutations at the given site on the branch above the given node, keeping
     * the mutations at all other sites. Branches without mutations at the site before and
     * after are left untouched.
     */
    private void replaceSiteMutations(Node node, int site, List<Mutation> newSiteMutations) {
        List<Mutation> branchMutations = this.mutations.getMutations(node);

        List<Mutation> newBranchMutations = new ArrayList<>();
        for (Mutation mutation : branchMutations) {
            if (mutation.site() != site) {
                newBranchMutations.add(mutation);
            }
        }

        if (newBranchMutations.size() == branchMutations.size() && newSiteMutations.isEmpty()) {
            return;
        }

        newBranchMutations.addAll(newSiteMutations);
        newBranchMutations.sort(Comparator.comparingDouble(Mutation::time).reversed());
        this.mutations.applyMutations(node, newBranchMutations, this);
    }

}
