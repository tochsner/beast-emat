package emat.operators.mutations;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;

import emat.helper.MissingSites;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.SiteHistorySampler;
import emat.stochasticmapping.SubstitutionProcess;
import emat.stochasticmapping.UniformisedStochasticMapping;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Deque;
import java.util.List;

@Description("Resamples the history of the clades whose data is missing around a tip, together with the state of " +
        "the node they attach to. A tip with missing sites is picked uniformly, and for each of its missing sites, the " +
        "clade grows upwards from the tip as long as all tips below are missing at the site, for at most maxLevels " +
        "levels and never up to the root. At the sites of a clade, the state of the node it attaches to, the paths of " +
        "the branches at this node and the history of the clade are drawn jointly from their exact conditional " +
        "distribution, given the states at the parent of the node and at its other children, which stay fixed. The " +
        "clades only depend on the tree and the data, which the move does not change, so the move is a Gibbs move that " +
        "is always accepted. Missations are not supported.")
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
    SiteHistorySampler sampler;
    int maxLevels;

    // the numbers of the tips with at least one missing site, which are the candidates of the move
    int[] missingTipNrs;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.maxLevels = this.maxLevelsInput.get();

        if (this.maxLevels < 0) {
            throw new IllegalArgumentException("The maximum number of levels must not be negative.");
        }

        int numStates = this.mutations.getAlignment().getMaxStateCount();
        SubstitutionProcess process = new SubstitutionProcess(this.geneticPriorInput.get(), this.tree, numStates);
        this.sampler = new SiteHistorySampler(this.mutations, process, new UniformisedStochasticMapping(numStates));
        this.missingSites = new MissingSites(this.mutations, this.tree);

        // the tips keep their numbers and data when the tree changes, but not their node objects
        this.missingTipNrs = this.tree.getExternalNodes().stream()
                .filter(tip -> this.missingSites.getSites(tip).length > 0)
                .mapToInt(Node::getNr)
                .toArray();
    }

    /**
     * Picks a tip with missing sites, collects the missing clades around it, and resamples
     * each of them with the node it attaches to. The regions are nested, as their tops lie on
     * the path from the tip upwards, but have disjoint sites, so each draw only touches the
     * sites of its own region and does not depend on the draws of the others.
     */
    @Override
    public double proposal() {
        if (this.missingTipNrs.length == 0 || this.tree.getInternalNodeCount() == 0) {
            return Double.NEGATIVE_INFINITY;
        }

        Node tip = this.tree.getNode(this.missingTipNrs[Randomizer.nextInt(this.missingTipNrs.length)]);
        for (Region region : this.collectRegions(tip)) {
            this.resampleRegion(region);
        }

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

    /* Resampling */

    /**
     * Replaces the history of the given region at its sites by a draw from its exact
     * conditional distribution. The region of the sampler has the node the clade attaches to
     * at the top, the other children of this node on the boundary, and the clade as its
     * data-free clade, as the clade is missing at these sites.
     */
    private void resampleRegion(Region region) {
        Node top = region.top();
        this.sampler.setUpRegion(top.getParent(), node -> false, top);

        List<List<Mutation>> newMutations = this.sampler.createEmptyHistory();
        BitSet sites = region.sites();
        for (int site = sites.nextSetBit(0); site >= 0; site = sites.nextSetBit(site + 1)) {
            this.sampler.sampleRegionSiteHistory(site, newMutations);
        }

        SiteMutations.replaceRegion(this.mutations, this.sampler, sites::get, newMutations, this);
    }

}
