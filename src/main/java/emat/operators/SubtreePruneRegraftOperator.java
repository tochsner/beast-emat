package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import emat.helper.BranchMutations;
import emat.stochasticmapping.ExactBranchHistories;
import emat.stochasticmapping.HybridBranchHistories;
import emat.stochasticmapping.JukesCantorStochasticMapping;
import emat.state.Mutation;
import emat.helper.MissingSites;
import emat.helper.MutationPaths;
import emat.helper.NodeStateLookup;
import emat.helper.SiteChanges;
import emat.helper.SiteStates;
import emat.helper.StarHistories;
import emat.state.Mutations;
import emat.prior.GeneticPrior;

import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntUnaryOperator;

@Description("Prunes a subtree and regrafts it elsewhere without changing the root (docs/mcmc-moves.md §3). " +
        "By default, only the history on the branch above the pruned subtree is resampled, by approximate " +
        "Jukes-Cantor stochastic mapping (docs/mcmc-moves.md §3.4) or, optionally, exactly under the model of the " +
        "genetic prior. Optionally, the histories on all branches " +
        "around the old and the new attachment point are resampled (docs/mcmc-moves.md §3.5). If the pruned " +
        "subtree is a tip, its states at the sites where its data is missing are resampled as well, so that " +
        "they follow its placement, and by default also among the compatible states where its data is partially ambiguous. Missations are not supported. Subclasses choose the pruned subtree and the " +
        "grafting point.")
public abstract class SubtreePruneRegraftOperator extends TreeOperator {

    public enum HistorySampler {
        JUKES_CANTOR, EXACT, HYBRID
    }

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<Boolean> resampleNeighbourhoodInput = new Input<>("resampleNeighbourhood", "whether to resample the histories on all branches around the old and the new attachment point instead of only on the branch above the pruned subtree", false);
    final public Input<HistorySampler> historySamplerInput = new Input<>("historySampler", "how the history on the branch above the pruned subtree is resampled: JUKES_CANTOR under an approximating Jukes-Cantor model, " +
            "EXACT from its conditional distribution under the model of the genetic prior, site by site, " +
            "HYBRID by choosing the sites with mutations under Jukes-Cantor and then sampling their histories exactly. EXACT and HYBRID are not supported together with resampleNeighbourhood", HistorySampler.JUKES_CANTOR, HistorySampler.values());
    final public Input<Boolean> resampleAmbiguousSitesInput = new Input<>("resampleAmbiguousSites", "whether a pruned tip also resamples its states among the compatible states at the sites where its data is partially ambiguous, " +
            "so that they follow its placement. Otherwise, these states stay fixed. Only the history on the branch above the pruned subtree supports it, so it has no effect with resampleNeighbourhood", true);

    // the columns of the star states: the outer ends G, S and X of a star around P, and its centre P
    static final int TOP = StarHistories.TOP;
    static final int SIBLING = StarHistories.FIRST;
    static final int SUBTREE = StarHistories.SECOND;
    static final int CENTRE = StarHistories.CENTRE;

    private static final int[] NO_SITES = new int[0];
    private static final BitSet NO_SITE_SET = new BitSet();

    Mutations mutations;
    GeneticPrior geneticPrior;
    Tree tree;
    JukesCantorStochasticMapping stochasticMapping;
    ExactBranchHistories exactBranchHistories;
    HybridBranchHistories hybridBranchHistories;
    StarHistories starHistories;
    BranchMutations branchMutations;
    boolean resampleNeighbourhood;

    // reusable site maps for the changes along the paths of the current proposal and the resulting differing sites
    SiteChanges subtreeChanges;
    SiteChanges newParentChanges;
    SiteChanges differingSites;
    SiteChanges combinedChanges;

    // reusable states of the stars around the old and the new attachment point at the sites where not all of their states agree
    SiteStates oldStarStates;
    SiteStates newStarStates;

    // reusable lookup of the states at X for the sites whose end states agree
    NodeStateLookup subtreeStates;

    // reusable lookup of the states at the old parent P, which start the old history on the branch above X
    NodeStateLookup parentStates;

    // per node: the sites where the data of a tip is missing, which are empty for inner nodes
    MissingSites missingSites;

    // the missing sites of X in the current proposal, whose states at X are resampled
    int[] subtreeMissingSites;
    BitSet subtreeMissingSiteSet;

    // whether the partially ambiguous sites of a pruned tip are resampled, and those of X in the current proposal with the bit masks of their compatible states, which are empty otherwise
    boolean resampleAmbiguousSites;
    int[] subtreeAmbiguousSites;
    BitSet subtreeAmbiguousSiteSet;
    int[] subtreeCompatibleStates;

    // the model of the current proposal on the branch above X: the total mutation rate λ(R) of the root sequence times the branch rate, and the fictitious Jukes-Cantor rate μ̃
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
        this.resampleNeighbourhood = this.resampleNeighbourhoodInput.get();
        HistorySampler historySampler = this.historySamplerInput.get();
        if (historySampler != HistorySampler.JUKES_CANTOR && this.resampleNeighbourhood) {
            throw new IllegalArgumentException("The " + historySampler + " history sampler does not support resampleNeighbourhood.");
        }
        int numStates = this.mutations.getAlignment().getMaxStateCount();
        if (historySampler == HistorySampler.EXACT) {
            this.exactBranchHistories = new ExactBranchHistories(numStates, numSites);
        } else if (historySampler == HistorySampler.HYBRID) {
            this.hybridBranchHistories = new HybridBranchHistories(numStates, numSites);
        }

        this.subtreeChanges = new SiteChanges(numSites);
        this.newParentChanges = new SiteChanges(numSites);
        this.differingSites = new SiteChanges(numSites);
        this.combinedChanges = new SiteChanges(numSites);
        this.starHistories = new StarHistories(this.stochasticMapping, numSites);
        this.oldStarStates = new SiteStates(numSites, StarHistories.NUM_COLUMNS);
        this.newStarStates = new SiteStates(numSites, StarHistories.NUM_COLUMNS);
        this.subtreeStates = new NodeStateLookup(this.mutations, numSites);
        this.parentStates = new NodeStateLookup(this.mutations, numSites);

        this.missingSites = new MissingSites(this.mutations, this.tree);
        this.resampleAmbiguousSites = this.resampleAmbiguousSitesInput.get() && !this.resampleNeighbourhood;
    }

    /**
     * Detaches the subtree below X from its parent P and regrafts it with P on the branch
     * above S' at height t_P'. Neither P nor P' is the root. If S' is the old sibling S, the
     * topology stays the same and only the height of P shifts along the joined G–S branch.
     * The new histories are proposed on the old tree, which is then regrafted.
     */
    @Override
    public double proposal() {
        Node x = this.pickSubtreeRoot();
        if (x == null) {
            return Double.NEGATIVE_INFINITY;
        }

        double logForwardPickProbability = this.computeLogSubtreeRootProbability(x);

        this.updateModel(x);
        this.subtreeMissingSites = this.missingSites.getSites(x);
        this.subtreeMissingSiteSet = this.missingSites.getSiteSet(x);
        this.subtreeAmbiguousSites = this.resampleAmbiguousSites ? this.missingSites.getAmbiguousSites(x) : NO_SITES;
        this.subtreeAmbiguousSiteSet = this.resampleAmbiguousSites ? this.missingSites.getAmbiguousSiteSet(x) : NO_SITE_SET;
        this.subtreeCompatibleStates = this.resampleAmbiguousSites ? this.missingSites.getCompatibleStates(x) : NO_SITES;

        // choose a grafting point

        GraftingPoint graftingPoint = this.proposeGraftingPoint(x);
        if (graftingPoint == null || !this.isValidGraftingPoint(x, graftingPoint)) {
            return Double.NEGATIVE_INFINITY;
        }

        // resample the local histories, then regraft and apply them

        Map<Node, List<Mutation>> newBranchMutations = new LinkedHashMap<>();
        double logMutationHastingsRatio = this.resampleNeighbourhood
                ? this.proposeNeighbourhoodHistories(x, graftingPoint, newBranchMutations)
                : this.proposeSubtreeHistory(x, graftingPoint, newBranchMutations);

        this.regraft(x, graftingPoint.newSibling(), graftingPoint.newParentHeight());

        for (Map.Entry<Node, List<Mutation>> entry : newBranchMutations.entrySet()) {
            this.mutations.applyMutations(entry.getKey(), entry.getValue(), this);
        }

        double logBackwardPickProbability = this.computeLogSubtreeRootProbability(x);

        return graftingPoint.logHastingsRatio() + logMutationHastingsRatio + logBackwardPickProbability - logForwardPickProbability;
    }

    /* Grafting Point Proposals */

    /**
     * Picks the root X of the subtree to prune, such that neither X nor its parent is the
     * root. Returns null if there is no such node. If the probability of the choice changes
     * under the move, computeLogSubtreeRootProbability has to return it.
     */
    protected abstract Node pickSubtreeRoot();

    /**
     * Computes the log probability that pickSubtreeRoot picks the given node in the current
     * state, up to a constant. It is evaluated before and after the move, and the difference
     * enters the Hastings ratio. A choice whose probability does not change under the move
     * needs no correction.
     */
    protected double computeLogSubtreeRootProbability(Node x) {
        return 0.0;
    }

    /**
     * Proposes the new attachment point of the subtree below X in the pruned tree, i.e. with
     * X and its parent P removed and P's two branches joined. Returns null if there is no
     * valid proposal. Candidates are the subtree slide, Wilson–Balding and mdSPR proposals of
     * docs/mcmc-moves.md §8.
     */
    protected abstract GraftingPoint proposeGraftingPoint(Node x);

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

    /* Grafting and Resampling */

    /**
     * Proposes the histories for a regraft that resamples only the
     * history on the new P'–X branch. The mutations of the old P branch are merged into the
     * branch of the old sibling S, the branch above S' is split at t_P', and everything else
     * is kept. At the missing sites of X, the states at X are first sampled from the states
     * at P' under Jukes-Cantor, and likewise among the compatible states at the partially
     * ambiguous sites if they are resampled. The history is then sampled given the end states. The
     * new mutations are put into the given map in the order in which they must be applied
     * after the regraft. Returns the log of α_mut(n → o) / α_mut(o → n).
     */
    private double proposeSubtreeHistory(Node x, GraftingPoint graftingPoint, Map<Node, List<Mutation>> newBranchMutations) {
        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);
        Node grandparent = parent.getParent();

        Node newSibling = graftingPoint.newSibling();
        boolean isHeightShift = newSibling == sibling;
        double newParentHeight = graftingPoint.newParentHeight();
        double oldParentHeight = parent.getHeight();

        // find the sites that differ between the ends of the old P–X branch, which are exactly the changes on it

        List<Mutation> oldSubtreeMutations = this.mutations.getMutations(x);
        MutationPaths.collectChanges(this.mutations, parent, x, x.getHeight(), this.subtreeChanges);
        int oldNumDifferingSites = MutationPaths.countDifferingSites(this.subtreeChanges);
        int oldNumDifferingMissingSites = this.countDifferingMissingSites(this.subtreeChanges);

        // P' lies on the branch above this node in the current tree, which is the old P branch if P shifts upwards

        Node newParentBranchNode = isHeightShift && newParentHeight > oldParentHeight ? parent : newSibling;

        // find the MRCA of X and P', then the changes from it down to both, which the new P'–X branch has to compensate

        Node mrca = MutationPaths.findMrca(x, newParentBranchNode.getParent());
        MutationPaths.collectChanges(this.mutations, mrca, x, x.getHeight(), this.subtreeChanges);
        MutationPaths.collectChanges(this.mutations, mrca, newParentBranchNode, newParentHeight, this.newParentChanges);
        MutationPaths.combineChanges(this.newParentChanges, this.subtreeChanges, this.combinedChanges);

        // keep the differing sites where the data of X is observed, and sample the states of X at its missing and its ambiguous sites

        this.differingSites.clear();
        for (int slot = 0; slot < this.combinedChanges.getSize(); slot++) {
            int site = this.combinedChanges.getSite(slot);
            if (!this.subtreeMissingSiteSet.get(site) && !this.subtreeAmbiguousSiteSet.get(site)) {
                this.differingSites.addSite(site, this.combinedChanges.getStartState(slot), this.combinedChanges.getEndState(slot));
            }
        }

        this.subtreeStates.resetFor(x);
        IntUnaryOperator newParentStates = this::getNewParentState;

        double newExpectedJumps = this.jukesCantorRate * (newParentHeight - x.getHeight());
        int newNumDifferingMissingSites = this.stochasticMapping.sampleFreeEndStates(
                newExpectedJumps, this.subtreeMissingSites, newParentStates, this.differingSites
        );
        double logAmbiguousForwardDensity = this.stochasticMapping.sampleCompatibleEndStates(
                newExpectedJumps, this.subtreeAmbiguousSites, this.subtreeCompatibleStates, newParentStates, this.differingSites
        );

        // sample the history on the new P'–X branch while the tree still holds the old sequences, where every agreeing site is in the state of P'

        List<Mutation> newSubtreeMutations;
        if (this.exactBranchHistories != null) {
            newSubtreeMutations = this.sampleExactBranchHistory(x, newParentHeight, newParentStates);
        } else if (this.hybridBranchHistories != null) {
            newSubtreeMutations = this.sampleHybridBranchHistory(x, newParentHeight, newParentStates);
        } else {
            newSubtreeMutations = this.stochasticMapping.sampleBranchHistory(
                    x.getNr(), newParentHeight, x.getHeight(), this.jukesCantorRate, this.differingSites, newParentStates
            );
        }

        // rearrange the mutations: join G–P–S into G–S, then split G'–S' into G'–P'–S'

        List<Mutation> joinedMutations = this.branchMutations.joinBranches(
                this.mutations.getMutations(parent), this.mutations.getMutations(sibling), sibling.getNr(), grandparent.getHeight()
        );
        newBranchMutations.put(sibling, joinedMutations);

        // if P only shifts, the split replaces the joined mutations of S, which keeps S first in the order
        List<Mutation> newSiblingMutations = isHeightShift ? joinedMutations : this.mutations.getMutations(newSibling);
        BranchMutations.Split split = this.branchMutations.splitBranch(newSiblingMutations, newParentHeight, parent.getNr(), newSibling.getNr());
        newBranchMutations.put(parent, split.upperMutations());
        newBranchMutations.put(newSibling, split.lowerMutations());

        newBranchMutations.put(x, newSubtreeMutations);

        // calculate the hastings correction

        double oldExpectedJumps = this.jukesCantorRate * (oldParentHeight - x.getHeight());

        double logForwardDensity = this.stochasticMapping.computeLogFreeEndStatesDensity(newExpectedJumps, this.subtreeMissingSites.length, newNumDifferingMissingSites);
        double logBackwardDensity = this.stochasticMapping.computeLogFreeEndStatesDensity(oldExpectedJumps, this.subtreeMissingSites.length, oldNumDifferingMissingSites);

        // the reverse move samples the old states of X at its ambiguous sites from the states at the old P, which the tree still holds
        logForwardDensity += logAmbiguousForwardDensity;
        logBackwardDensity += this.stochasticMapping.computeLogCompatibleEndStatesDensity(
                oldExpectedJumps, this.subtreeAmbiguousSites, this.subtreeCompatibleStates, this.parentStates.resetFor(parent), this.subtreeStates
        );

        if (this.hybridBranchHistories != null) {
            double rateScale = this.computeExactRateScale(x);
            logForwardDensity += this.hybridBranchHistories.computeLogBranchHistoryDensity(
                    newSubtreeMutations, newParentHeight, x.getHeight(), rateScale, this.jukesCantorRate, this.differingSites.getSize()
            );
            logBackwardDensity += this.hybridBranchHistories.computeLogBranchHistoryDensity(
                    oldSubtreeMutations, oldParentHeight, x.getHeight(), rateScale, this.jukesCantorRate, oldNumDifferingSites
            );
        } else if (this.exactBranchHistories == null) {
            logForwardDensity += this.computeLogBranchHistoryDensity(x, newParentHeight, newSubtreeMutations, this.differingSites.getSize());
            logBackwardDensity += this.computeLogBranchHistoryDensity(x, oldParentHeight, oldSubtreeMutations, oldNumDifferingSites);
        } else {
            double rateScale = this.computeExactRateScale(x);
            logForwardDensity += this.exactBranchHistories.computeLogBranchHistoryDensity(
                    newSubtreeMutations, newParentHeight, x.getHeight(), rateScale, newParentStates
            );
            logBackwardDensity += this.exactBranchHistories.computeLogBranchHistoryDensity(
                    oldSubtreeMutations, oldParentHeight, x.getHeight(), rateScale, this.parentStates.resetFor(parent)
            );
        }

        return logBackwardDensity - logForwardDensity;
    }

    /**
     * Proposes the histories for a regraft that resamples the histories
     * on all branches around the old and the new attachment point (docs/mcmc-moves.md §3.5). These are the branches above P,
     * S, X and S', which form the star G–P, P–S, P–X and the branch G'–S' before the move,
     * and the joined branch G–S and the star G'–P', P'–S', P'–X after it. The sequences at
     * the outer ends G, S, X, G' and S' stay fixed, except at the missing sites of X. The
     * new histories are proposed under Jukes-Cantor: first the states at P' given the outer
     * ends of its star, then the states of X at its missing sites given P', then the history
     * on every branch given its end states. The reverse move proposes the old
     * histories in the same way, and the genetic prior corrects for the approximate model.
     * The new mutations are put into the given map in the order in which they must be
     * applied after the regraft. Returns the log of α_mut(n → o) / α_mut(o → n).
     */
    private double proposeNeighbourhoodHistories(Node x, GraftingPoint graftingPoint, Map<Node, List<Mutation>> newBranchMutations) {
        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);
        Node grandparent = parent.getParent();

        Node newSibling = graftingPoint.newSibling();
        Node newGrandparent = this.getPrunedParent(x, newSibling);
        boolean isHeightShift = newSibling == sibling;
        double newParentHeight = graftingPoint.newParentHeight();
        double oldParentHeight = parent.getHeight();

        // the fictitious Jukes-Cantor rate of a branch is its branch rate times λ(R) / L, which the move keeps as the branches keep the rates of the nodes below them

        double ratePerSite = this.geneticPrior.getTotalMutationRate(this.tree.getRoot()) / this.mutations.getReferenceSequence().length;
        double parentRate = this.geneticPrior.branchRateModel.getRateForBranch(parent) * ratePerSite;
        double siblingRate = this.geneticPrior.branchRateModel.getRateForBranch(sibling) * ratePerSite;
        double newSiblingRate = this.geneticPrior.branchRateModel.getRateForBranch(newSibling) * ratePerSite;
        double subtreeRate = this.jukesCantorRate;

        List<Mutation> oldParentMutations = this.mutations.getMutations(parent);
        List<Mutation> oldSiblingMutations = this.mutations.getMutations(sibling);
        List<Mutation> oldSubtreeMutations = this.mutations.getMutations(x);
        List<Mutation> oldNewSiblingMutations = this.mutations.getMutations(newSibling);

        // find the states at the outer ends of both stars, and at the centre P of the old star

        this.starHistories.collectOuterStates(this.mutations::getMutations, grandparent, sibling, x, this.oldStarStates);
        this.starHistories.collectCentreStates(oldParentMutations, this.oldStarStates);
        this.starHistories.collectOuterStates(this.mutations::getMutations, newGrandparent, newSibling, x, this.newStarStates);

        // sample the new histories while the tree still holds the sequences at the outer ends, which the move keeps

        this.subtreeStates.resetFor(x);

        double[] newExpectedJumps = {
                parentRate * (newGrandparent.getHeight() - newParentHeight),
                newSiblingRate * (newParentHeight - newSibling.getHeight()),
                subtreeRate * (newParentHeight - x.getHeight())
        };
        double logForwardDensity = this.stochasticMapping.sampleStarStates(
                newExpectedJumps, this.newStarStates, this.subtreeStates, this.subtreeMissingSites, this.subtreeMissingSiteSet, SUBTREE
        );

        List<Mutation> newParentMutations = this.starHistories.sampleBranchHistory(
                parent, newGrandparent.getHeight(), newParentHeight, parentRate, this.newStarStates, TOP, CENTRE, this.subtreeStates
        );
        List<Mutation> newSiblingMutations = this.starHistories.sampleBranchHistory(
                newSibling, newParentHeight, newSibling.getHeight(), newSiblingRate, this.newStarStates, CENTRE, SIBLING, this.subtreeStates
        );
        List<Mutation> newSubtreeMutations = this.starHistories.sampleBranchHistory(
                x, newParentHeight, x.getHeight(), subtreeRate, this.newStarStates, CENTRE, SUBTREE, this.subtreeStates
        );

        logForwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                newParentMutations, newGrandparent.getHeight(), newParentHeight, parentRate, this.newStarStates, TOP, CENTRE
        );
        logForwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                newSiblingMutations, newParentHeight, newSibling.getHeight(), newSiblingRate, this.newStarStates, CENTRE, SIBLING
        );
        logForwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                newSubtreeMutations, newParentHeight, x.getHeight(), subtreeRate, this.newStarStates, CENTRE, SUBTREE
        );

        // the old star around P, which the reverse move resamples

        double[] oldExpectedJumps = {
                parentRate * (grandparent.getHeight() - oldParentHeight),
                siblingRate * (oldParentHeight - sibling.getHeight()),
                subtreeRate * (oldParentHeight - x.getHeight())
        };
        double logBackwardDensity = this.stochasticMapping.computeLogStarStatesDensity(
                oldExpectedJumps, this.oldStarStates, this.subtreeMissingSites.length, this.subtreeMissingSiteSet, SUBTREE
        );

        logBackwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                oldParentMutations, grandparent.getHeight(), oldParentHeight, parentRate, this.oldStarStates, TOP, CENTRE
        );
        logBackwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                oldSiblingMutations, oldParentHeight, sibling.getHeight(), siblingRate, this.oldStarStates, CENTRE, SIBLING
        );
        logBackwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                oldSubtreeMutations, oldParentHeight, x.getHeight(), subtreeRate, this.oldStarStates, CENTRE, SUBTREE
        );

        // unless P only shifts, the move also joins G–S, and the reverse move joins G'–S'

        List<Mutation> joinedMutations = null;
        if (!isHeightShift) {
            joinedMutations = this.starHistories.sampleBranchHistory(
                    sibling, grandparent.getHeight(), sibling.getHeight(), siblingRate, this.oldStarStates, TOP, SIBLING, this.subtreeStates
            );
            logForwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                    joinedMutations, grandparent.getHeight(), sibling.getHeight(), siblingRate, this.oldStarStates, TOP, SIBLING
            );
            logBackwardDensity += this.starHistories.computeLogBranchHistoryDensity(
                    oldNewSiblingMutations, newGrandparent.getHeight(), newSibling.getHeight(), newSiblingRate, this.newStarStates, TOP, SIBLING
            );
        }

        newBranchMutations.put(parent, newParentMutations);
        newBranchMutations.put(newSibling, newSiblingMutations);
        newBranchMutations.put(x, newSubtreeMutations);
        if (!isHeightShift) {
            newBranchMutations.put(sibling, joinedMutations);
        }

        return logBackwardDensity - logForwardDensity;
    }

    /* Stochastic Mapping */

    /**
     * Takes the current model from the genetic prior. The fictitious Jukes-Cantor rate is
     * μ̃ = λ(R) / L, with λ(R) the total mutation rate of the root sequence times the branch
     * rate of X, which the genetic prior has cached for the current state. The move keeps
     * the root sequence and the branch rate of X, so the reverse move uses the same model,
     * even though the sequence of X changes at its missing sites.
     */
    private void updateModel(Node x) {
        double branchRate = this.geneticPrior.branchRateModel.getRateForBranch(x);
        int numSites = this.mutations.getReferenceSequence().length;

        this.subtreeTotalRate = branchRate * this.geneticPrior.getTotalMutationRate(this.tree.getRoot());
        this.jukesCantorRate = this.subtreeTotalRate / numSites;
    }

    /**
     * Returns the state at P' of the current proposal. The changes from the MRCA of X and P'
     * down to both must be collected, and the lookup of the states at X be reset.
     */
    private int getNewParentState(int site) {
        int slot = this.newParentChanges.getSlot(site);
        if (slot >= 0) {
            return this.newParentChanges.getEndState(slot);
        }

        // without changes towards P', P' keeps the state at the MRCA, which is also the state of X unless it changes towards X
        slot = this.subtreeChanges.getSlot(site);
        return slot >= 0 ? this.subtreeChanges.getStartState(slot) : this.subtreeStates.applyAsInt(site);
    }

    /** Counts the missing sites of X whose states differ between the start and the end of the given changes. */
    private int countDifferingMissingSites(SiteChanges changes) {
        int numDifferingMissingSites = 0;
        for (int slot = 0; slot < changes.getSize(); slot++) {
            if (changes.getStartState(slot) != changes.getEndState(slot) && this.subtreeMissingSiteSet.get(changes.getSite(slot))) {
                numDifferingMissingSites++;
            }
        }
        return numDifferingMissingSites;
    }

    /**
     * Samples the history on the new P'-X branch exactly under the model of the genetic prior,
     * given the differing sites and, at every other site, the state at P'.
     */
    private List<Mutation> sampleExactBranchHistory(Node x, double newParentHeight, IntUnaryOperator newParentStates) {
        Node root = this.tree.getRoot();
        this.exactBranchHistories.setModel(this.geneticPrior.computeRateMatrix(), this.geneticPrior.substitutionModel.getEigenDecomposition(root));

        return this.exactBranchHistories.sampleBranchHistory(
                x.getNr(), newParentHeight, x.getHeight(), this.computeExactRateScale(x), this.differingSites, newParentStates
        );
    }

    /**
     * Samples the history on the new P'-X branch with the sites chosen under Jukes-Cantor with
     * the rate μ̃ and their histories drawn exactly under the model of the genetic prior.
     */
    private List<Mutation> sampleHybridBranchHistory(Node x, double newParentHeight, IntUnaryOperator newParentStates) {
        Node root = this.tree.getRoot();
        this.hybridBranchHistories.setModel(this.geneticPrior.computeRateMatrix(), this.geneticPrior.substitutionModel.getEigenDecomposition(root));

        return this.hybridBranchHistories.sampleBranchHistory(
                x.getNr(), newParentHeight, x.getHeight(), this.computeExactRateScale(x), this.jukesCantorRate, this.differingSites, newParentStates
        );
    }

    /** Returns the scale s of the rate matrix Q on the branch above X: its branch rate times the site rate. The move keeps both. */
    private double computeExactRateScale(Node x) {
        return this.geneticPrior.branchRateModel.getRateForBranch(x) * this.geneticPrior.siteModel.getRateForCategory(0, this.tree.getRoot());
    }

    /**
     * Computes the log probability α_mut that the Jukes-Cantor mapping with the rate μ̃
     * proposes the given history on the branch above X that starts at the given height,
     * given the number of sites whose states differ between its ends.
     */
    protected double computeLogBranchHistoryDensity(Node x, double startHeight, List<Mutation> branchMutations, int numDifferingSites) {
        return this.stochasticMapping.computeLogBranchHistoryDensity(
                branchMutations, startHeight, x.getHeight(), this.jukesCantorRate, numDifferingSites
        );
    }

    /* Tree Rearrangement */

    /**
     * Moves P with the subtree below X onto the branch above S' at the given height. If S'
     * is the old sibling, only the height of P changes.
     */
    private void regraft(Node x, Node newSibling, double newParentHeight) {
        Node parent = x.getParent();
        Node sibling = this.getOtherChild(parent, x);
        Node grandparent = parent.getParent();
        Node newGrandparent = this.getPrunedParent(x, newSibling);

        if (newSibling != sibling) {
            this.replace(grandparent, parent, sibling);
            this.replace(parent, sibling, newSibling);
            this.replace(newGrandparent, newSibling, parent);
        }
        parent.setHeight(newParentHeight);
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
