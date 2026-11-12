package esat.operators.states;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import esat.helper.TipCompatibility;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import esat.stochasticmapping.MissingCladeStateSampler;
import esat.stochasticmapping.SubstitutionProcess;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Deque;
import java.util.List;

@Description("Resamples the states of the clades whose data is missing around a tip, together with the state of " +
        "the node they attach to. A tip with missing sites is picked uniformly, and for each of its missing sites, the " +
        "clade grows upwards from the tip as long as all tips below are missing at the site, for at most maxLevels " +
        "levels and never up to the root. At the sites of a clade, the state of the node it attaches to and the states " +
        "of the clade are drawn jointly from their exact conditional distribution, given the states at the parent of " +
        "the node and at its other children, which stay fixed. The clades only depend on the tree and the data, which " +
        "the move does not change, so the move is a Gibbs move that is always accepted. If a clade has more than " +
        "maxSites sites, a uniformly random subset of maxSites of them is resampled, which bounds the cost of a " +
        "proposal and is exact as the sites are independent given the tree.")
public class MissingCladeStateGibbsOperator extends Operator {

    final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "the node states to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Integer> maxLevelsInput = new Input<>("maxLevels", "the maximum number of levels a clade grows above the picked tip. With 0, only the branch above the tip is resampled", Integer.MAX_VALUE);
    final public Input<Integer> maxSitesInput = new Input<>("maxSites", "the maximum number of sites resampled per clade, drawn uniformly without replacement from its sites if there are more", Integer.MAX_VALUE);

    /** The top of a clade and the sites at which its states are resampled. */
    private record Region(Node top, BitSet sites) {
    }

    NodeStates nodeStates;
    TreeInterface tree;
    SubstitutionProcess substitutionProcess;
    MissingCladeStateSampler sampler;
    int maxLevels;
    int maxSites;

    // per node number: the missing sites of a tip
    BitSet[] missingSiteSetOfNode;

    // the numbers of the tips with at least one missing site, which are the candidates of the move
    int[] missingTipNrs;

    @Override
    public void initAndValidate() {
        this.nodeStates = this.nodeStatesInput.get();
        this.tree = this.nodeStates.getTree();
        this.maxLevels = this.maxLevelsInput.get();
        this.maxSites = this.maxSitesInput.get();

        if (this.maxLevels < 0) {
            throw new IllegalArgumentException("The maximum number of levels must not be negative.");
        }
        if (this.maxSites < 1) {
            throw new IllegalArgumentException("The maximum number of sites must be positive.");
        }

        this.substitutionProcess = new SubstitutionProcess(this.geneticPriorInput.get());
        this.sampler = new MissingCladeStateSampler(this.nodeStates, this.substitutionProcess);

        // the tips keep their numbers and data when the tree changes, but not their node objects
        TipCompatibility tipCompatibility = this.substitutionProcess.getTipCompatibility();
        this.missingSiteSetOfNode = new BitSet[this.tree.getNodeCount()];
        List<Integer> tipNrs = new ArrayList<>();
        for (Node tip : this.tree.getExternalNodes()) {
            BitSet missingSites = new BitSet(this.nodeStates.getNumSites());
            for (int site : tipCompatibility.getMissingSites(tip)) {
                missingSites.set(site);
            }
            this.missingSiteSetOfNode[tip.getNr()] = missingSites;
            if (!missingSites.isEmpty()) {
                tipNrs.add(tip.getNr());
            }
        }
        this.missingTipNrs = tipNrs.stream().mapToInt(Integer::intValue).toArray();
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

        this.substitutionProcess.update();
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
        BitSet remainingSites = (BitSet) this.missingSiteSetOfNode[tip.getNr()].clone();

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
                sites.and(this.missingSiteSetOfNode[next.getNr()]);
            } else {
                for (Node child : next.getChildren()) {
                    stack.push(child);
                }
            }
        }
    }

    /* Resampling */

    /**
     * Replaces the states of the attach node of the given region and of its clade at the sites
     * of the region, or at a uniformly random subset of at most maxSites of them, by a draw
     * from their exact conditional distribution. The new states of the attach node are set first,
     * which rewrites the changes of its children, then the changes of the clade are installed
     * from the top down, as the counts of a branch use the state counts of its parent.
     */
    private void resampleRegion(Region region) {
        int[] sites = this.chooseSites(region.sites()).stream().toArray();
        MissingCladeStateSampler.CladeSample sample = this.sampler.sample(region.top(), sites);

        this.nodeStates.setStates(region.top().getParent(), sites, sample.attachStates(), this);
        for (int i = 0; i < sample.cladeNodes().size(); i++) {
            this.nodeStates.setBranchChanges(sample.cladeNodes().get(i), sample.changesOfCladeNodes()[i], this);
        }
    }

    /**
     * Returns the given sites if there are at most maxSites of them, and otherwise a
     * uniformly random subset of maxSites of them, drawn by a partial shuffle.
     */
    private BitSet chooseSites(BitSet sites) {
        if (sites.cardinality() <= this.maxSites) {
            return sites;
        }

        int[] candidates = sites.stream().toArray();
        BitSet chosenSites = new BitSet();
        for (int i = 0; i < this.maxSites; i++) {
            int j = i + Randomizer.nextInt(candidates.length - i);
            int site = candidates[j];
            candidates[j] = candidates[i];
            candidates[i] = site;
            chosenSites.set(site);
        }
        return chosenSites;
    }

}
