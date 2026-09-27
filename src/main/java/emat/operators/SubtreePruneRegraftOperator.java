package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.helper.BranchMutations;
import emat.helper.JukesCantorStochasticMapping;
import emat.state.Mutation;
import emat.helper.MutationPaths;
import emat.helper.SiteChanges;
import emat.state.Mutations;
import emat.prior.GeneticPrior;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Description("Prunes a subtree and regrafts it elsewhere without changing the root (docs/mcmc-moves.md §3). " +
        "Only the history on the branch above the pruned subtree is resampled, by approximate Jukes-Cantor " +
        "stochastic mapping (docs/mcmc-moves.md §3.4). Missations are not supported. " +
        "Subclasses choose the pruned subtree and the grafting point.")
public abstract class SubtreePruneRegraftOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    Mutations mutations;
    GeneticPrior geneticPrior;
    Tree tree;
    JukesCantorStochasticMapping stochasticMapping;
    BranchMutations branchMutations;

    // reusable site maps for the changes along the paths of the current proposal and the resulting differing sites
    SiteChanges subtreeChanges;
    SiteChanges newParentChanges;
    SiteChanges differingSites;

    // the model of the current proposal on the branch above X: the total mutation rate λ(X) times the branch rate, and the fictitious Jukes-Cantor rate μ̃
    double subtreeTotalRate;
    double jukesCantorRate;

    /**
     * A new attachment point for the pruned subtree: the branch above the new sibling S' in
     * the pruned tree, the height of the new parent P' on it, and the log of
     * α_graft(n → o) / α_graft(o → n).
     */
    protected record GraftingPoint(Node newSibling, double newParentHeight, double logHastingsRatio) {
    }

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.geneticPrior = this.geneticPriorInput.get();
        this.tree = this.treeInput.get();

        int numSites = this.mutations.getReferenceSequence().length;
        this.stochasticMapping = new JukesCantorStochasticMapping(this.mutations.getAlignment().getMaxStateCount(), numSites);
        this.branchMutations = new BranchMutations(numSites);

        this.subtreeChanges = new SiteChanges(numSites);
        this.newParentChanges = new SiteChanges(numSites);
        this.differingSites = new SiteChanges(numSites);
    }

    /**
     * Detaches the subtree below X from its parent P and regrafts it with P on the branch
     * above S' at height t_P'. Neither P nor P' is the root. The mutations of the old P
     * branch are merged into the branch of the old sibling S, the branch above S' is split
     * at t_P', and the history on the new P'–X branch is resampled. Everything else is kept.
     * If S' is the old sibling S, the topology stays the same and only the height of P
     * shifts along the joined G–S branch.
     */
    @Override
    public double proposal() {
        Node x = this.pickSubtreeRoot();
        if (x == null) {
            return Double.NEGATIVE_INFINITY;
        }

        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);
        Node grandparent = parent.getParent();

        // the grafting strategy may use the model of the branch above X
        this.updateModel(x);

        GraftingPoint graftingPoint = this.proposeGraftingPoint(x);
        if (graftingPoint == null || !this.isValidGraftingPoint(x, graftingPoint)) {
            return Double.NEGATIVE_INFINITY;
        }

        Node newSibling = graftingPoint.newSibling();
        Node newGrandparent = this.getPrunedParent(x, newSibling);
        boolean isHeightShift = newSibling == sibling;
        double newParentHeight = graftingPoint.newParentHeight();
        double oldParentHeight = parent.getHeight();

        // find the sites that differ between the ends of the old P–X branch, which are exactly the changes on it

        List<Mutation> oldSubtreeMutations = this.mutations.getMutations(x);
        MutationPaths.collectChanges(this.mutations, parent, x, x.getHeight(), this.subtreeChanges);
        int oldNumDifferingSites = MutationPaths.countDifferingSites(this.subtreeChanges);

        // P' lies on the branch above this node in the current tree, which is the old P branch if P shifts upwards

        Node newParentBranchNode = isHeightShift && newParentHeight > oldParentHeight ? parent : newSibling;

        // find the MRCA of X and P', then the changes from it down to both, which the new P'–X branch has to compensate

        Node mrca = MutationPaths.findMrca(x, newParentBranchNode.getParent());
        MutationPaths.collectChanges(this.mutations, mrca, x, x.getHeight(), this.subtreeChanges);
        MutationPaths.collectChanges(this.mutations, mrca, newParentBranchNode, newParentHeight, this.newParentChanges);
        MutationPaths.combineChanges(this.newParentChanges, this.subtreeChanges, this.differingSites);

        // sample the history on the new P'–X branch while the tree still holds the sequence of X, which the move keeps

        List<Mutation> newSubtreeMutations = this.sampleBranchHistory(x, newParentHeight, this.differingSites);

        // rearrange the mutations: join G–P–S into G–S, then split G'–S' into G'–P'–S'

        Map<Node, List<Mutation>> newBranchMutations = new LinkedHashMap<>();

        List<Mutation> joinedMutations = this.branchMutations.joinBranches(
                this.mutations.getMutations(parent), this.mutations.getMutations(sibling), sibling.getNr(), grandparent.getHeight()
        );
        newBranchMutations.put(sibling, joinedMutations);

        List<Mutation> newSiblingMutations = isHeightShift ? joinedMutations : this.mutations.getMutations(newSibling);
        BranchMutations.Split split = this.branchMutations.splitBranch(newSiblingMutations, newParentHeight, parent.getNr(), newSibling.getNr());
        newBranchMutations.put(parent, split.upperMutations());
        newBranchMutations.put(newSibling, split.lowerMutations());

        // rearrange the tree, which only moves P for a height shift

        if (!isHeightShift) {
            this.replace(grandparent, parent, sibling);
            this.replace(parent, sibling, newSibling);
            this.replace(newGrandparent, newSibling, parent);
        }
        parent.setHeight(newParentHeight);

        for (Map.Entry<Node, List<Mutation>> entry : newBranchMutations.entrySet()) {
            this.mutations.applyMutations(entry.getKey(), entry.getValue(), this);
        }

        this.mutations.applyMutations(x, newSubtreeMutations, this);

        // combine the grafting and the mutation proposal probabilities

        double logForwardDensity = this.computeLogBranchHistoryDensity(x, newParentHeight, newSubtreeMutations, this.differingSites.getSize());
        double logBackwardDensity = this.computeLogBranchHistoryDensity(x, oldParentHeight, oldSubtreeMutations, oldNumDifferingSites);

        return graftingPoint.logHastingsRatio() + logBackwardDensity - logForwardDensity;
    }

    /* Grafting Strategy */

    /**
     * Picks the root X of the subtree to prune, such that neither X nor its parent is the
     * root. Returns null if there is no such node. The choice must be accounted for in the
     * Hastings ratio unless the set of candidates does not change under the move.
     */
    protected abstract Node pickSubtreeRoot();

    /**
     * Proposes the new attachment point of the subtree below X in the pruned tree, i.e. with
     * X and its parent P removed and P's two branches joined. Returns null if there is no
     * valid proposal. Candidates are the subtree slide, Wilson–Balding and mdSPR proposals of
     * docs/mcmc-moves.md §8.
     */
    protected abstract GraftingPoint proposeGraftingPoint(Node x);

    /* Stochastic Mapping */

    /**
     * Takes the current model from the genetic prior. The branch above X keeps its rate and
     * X keeps its sequence under the move, so the model of the old and the new P–X branch is
     * the same. The fictitious Jukes-Cantor rate is μ̃ = λ(X) / L, with λ(X) the total
     * mutation rate at X times the branch rate, which the genetic prior has cached for the
     * current state.
     */
    private void updateModel(Node x) {
        double branchRate = this.geneticPrior.branchRateModel.getRateForBranch(x);
        int numSites = this.mutations.getReferenceSequence().length;

        this.subtreeTotalRate = branchRate * this.geneticPrior.getTotalMutationRate(x);
        this.jukesCantorRate = this.subtreeTotalRate / numSites;
    }

    /**
     * Samples a new history on the branch above X that starts at the given height, given the
     * sites whose states differ between its ends. This maps every site under Jukes-Cantor
     * with the rate μ̃ (docs/mcmc-moves.md §4), so its cost scales with the number of
     * differing sites rather than the genome length, and the genetic prior corrects for the
     * approximate model in the acceptance probability. The returned mutations belong to X
     * and are sorted by descending height.
     */
    protected List<Mutation> sampleBranchHistory(Node x, double startHeight, SiteChanges differingSites) {
        return this.stochasticMapping.sampleBranchHistory(
                x.getNr(), startHeight, x.getHeight(), this.jukesCantorRate, differingSites,
                site -> MutationPaths.getState(this.mutations, x, site)
        );
    }

    /**
     * Computes the log probability α_mut that sampleBranchHistory proposes the given history
     * on the branch above X that starts at the given height, given the number of sites whose
     * states differ between its ends.
     */
    protected double computeLogBranchHistoryDensity(Node x, double startHeight, List<Mutation> branchMutations, int numDifferingSites) {
        return this.stochasticMapping.computeLogBranchHistoryDensity(
                branchMutations, startHeight, x.getHeight(), this.jukesCantorRate, numDifferingSites
        );
    }

    /* Tree Rearrangement */

    /**
     * Checks that the grafting point lies on a branch of the pruned tree that does not end
     * at the root, and that the new parent is older than X and the new sibling and younger
     * than the new grandparent.
     */
    private boolean isValidGraftingPoint(Node x, GraftingPoint graftingPoint) {
        Node parent = x.getParent();
        Node newSibling = graftingPoint.newSibling();
        double newParentHeight = graftingPoint.newParentHeight();

        if (newSibling == parent || newSibling.isRoot() || this.isInSubtree(newSibling, x)) {
            return false;
        }

        return newParentHeight > x.getHeight()
                && newParentHeight > newSibling.getHeight()
                && newParentHeight < this.getPrunedParent(x, newSibling).getHeight();
    }

    /**
     * Returns the parent of the given node in the pruned tree, where X and its parent P are
     * removed and the old sibling S hangs directly below the old grandparent G.
     */
    protected Node getPrunedParent(Node x, Node node) {
        Node parent = x.getParent();
        return node.getParent() == parent ? parent.getParent() : node.getParent();
    }

    /** Checks whether the given node lies in the subtree below the given root, including the root itself. */
    private boolean isInSubtree(Node node, Node subtreeRoot) {
        for (Node ancestor = node; ancestor != null; ancestor = ancestor.getParent()) {
            if (ancestor == subtreeRoot) {
                return true;
            }
        }
        return false;
    }

}
