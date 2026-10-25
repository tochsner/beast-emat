package emat.operators.mutations;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.helper.MutationPaths;
import emat.helper.NodeStateLookup;
import emat.helper.SiteChanges;
import emat.operators.topology.BranchMutations;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.BranchHistorySampler;
import emat.stochasticmapping.HistorySampler;
import emat.stochasticmapping.SubstitutionProcess;

import java.util.List;

@Description("Resamples the history of a branch without changing the tree (branch reform, docs/mcmc-moves.md §9.2). " +
        "On a branch below an internal node, the whole history is resampled by the chosen history sampler given the " +
        "sequences at both ends, so mutations can appear and disappear. On a branch below a binary root, the history " +
        "of the whole path through the root to the other child is resampled, which lets mutations move between the " +
        "two root branches and changes the root sequence. The branch is chosen uniformly or with a weight of its " +
        "number of mutations plus one. Missations are not supported.")
public class BranchReformOperator extends Operator {

    /** How the branch to resample is chosen. */
    public enum BranchSelection {
        UNIFORM, MUTATION_COUNT
    }

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<HistorySampler> historySamplerInput = new Input<>("historySampler", "how the history of a branch is resampled: JUKES_CANTOR under an approximating Jukes-Cantor model, " +
            "HYBRID by choosing the sites with mutations with one probability for all states and then sampling their histories exactly under the model of the genetic prior", HistorySampler.HYBRID, HistorySampler.values());
    final public Input<BranchSelection> branchSelectionInput = new Input<>("branchSelection", "how the branch is chosen: UNIFORM among all branches, or MUTATION_COUNT with a weight of the number of mutations " +
            "on the branch plus one, which resamples every mutation equally often", BranchSelection.MUTATION_COUNT, BranchSelection.values());

    Mutations mutations;
    TreeInterface tree;
    GeneticPrior geneticPrior;
    BranchSelection branchSelection;
    BranchMutations mutationLinker;
    SubstitutionProcess process;
    BranchHistorySampler historySampler;
    RootPathResampler rootPathResampler;

    // reusable map of the sites whose states differ between the ends of the branch, and lookup of the states at its end
    SiteChanges branchChanges;
    SiteChanges differingSites;
    NodeStateLookup endStates;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.geneticPrior = this.geneticPriorInput.get();
        this.branchSelection = this.branchSelectionInput.get();

        int numSites = this.mutations.getReferenceSequence().length;
        this.mutationLinker = new BranchMutations(numSites);
        this.process = new SubstitutionProcess(this.geneticPrior, this.tree, this.mutations.getAlignment().getMaxStateCount());
        this.historySampler = this.historySamplerInput.get().create(this.process, numSites);
        this.rootPathResampler = new RootPathResampler(this.mutations, this.geneticPrior, this.process, this.historySampler);

        this.branchChanges = new SiteChanges(numSites);
        this.differingSites = new SiteChanges(numSites);
        this.endStates = new NodeStateLookup(this.mutations, numSites);
    }

    /**
     * Chooses a branch and resamples its history. Both branches below a binary root resample the
     * same path, which keeps the root height. The probability of choosing the branch changes with
     * its number of mutations under MUTATION_COUNT, which enters the Hastings ratio.
     */
    @Override
    public double proposal() {
        if (this.tree.getNodeCount() < 2) {
            return Double.NEGATIVE_INFINITY;
        }

        double totalWeight = this.computeTotalWeight();
        Node node = this.selectBranch(totalWeight);
        double oldWeight = this.computeWeight(node);

        this.process.update();
        double logHastingsRatio = this.isOnRootPath(node)
                ? this.rootPathResampler.resample(node.getParent(), node.getParent().getHeight(), this)
                : this.resampleBranch(node);

        // only the weight of the chosen branch changes, so the new total follows from it

        double newWeight = this.computeWeight(node);
        double newTotalWeight = totalWeight - oldWeight + newWeight;
        return logHastingsRatio + Math.log(newWeight / newTotalWeight) - Math.log(oldWeight / totalWeight);
    }

    /* Branch Selection */

    /** Returns the sum of the weights of all branches, which is the normaliser of the branch choice. */
    private double computeTotalWeight() {
        double totalWeight = 0.0;
        int rootNr = this.tree.getRoot().getNr();
        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            if (nodeNr != rootNr) {
                totalWeight += this.computeBranchWeight(this.tree.getNode(nodeNr));
            }
        }
        return totalWeight;
    }

    /** Returns a branch below a non-root node, drawn with a probability proportional to its weight. */
    private Node selectBranch(double totalWeight) {
        int rootNr = this.tree.getRoot().getNr();

        if (this.branchSelection == BranchSelection.UNIFORM) {
            while (true) {
                int nodeNr = Randomizer.nextInt(this.tree.getNodeCount());
                if (nodeNr != rootNr) {
                    return this.tree.getNode(nodeNr);
                }
            }
        }

        double threshold = Randomizer.nextDouble() * totalWeight;
        Node lastNode = null;
        for (int nodeNr = 0; nodeNr < this.tree.getNodeCount(); nodeNr++) {
            if (nodeNr == rootNr) {
                continue;
            }

            lastNode = this.tree.getNode(nodeNr);
            threshold -= this.computeBranchWeight(lastNode);
            if (threshold < 0.0) {
                return lastNode;
            }
        }

        // rounding can leave a tiny remainder, which belongs to the last branch
        return lastNode;
    }

    /**
     * Returns the weight with which the move resamples the history of the given branch: its own
     * weight, or the weights of both root branches if the branch lies on the root path.
     */
    private double computeWeight(Node node) {
        if (!this.isOnRootPath(node)) {
            return this.computeBranchWeight(node);
        }

        double weight = 0.0;
        for (Node child : node.getParent().getChildren()) {
            weight += this.computeBranchWeight(child);
        }
        return weight;
    }

    /** Returns the weight of the branch above the given node: 1, or its number of mutations plus one. */
    private double computeBranchWeight(Node node) {
        if (this.branchSelection == BranchSelection.UNIFORM) {
            return 1.0;
        }
        return this.mutations.getMutations(node).size() + 1.0;
    }

    /** Returns whether the branch above the given node lies below a binary root, which the move resamples as one path. */
    private boolean isOnRootPath(Node node) {
        Node parent = node.getParent();
        return parent.isRoot() && parent.getChildCount() == 2;
    }

    /* Branches Below an Internal Node */

    /**
     * Resamples the history on the branch above the given node given the sequences at both of its
     * ends, which stay fixed. The Jukes-Cantor rate μ̃ = r λ(N) / L only depends on the branch
     * rate r and the sequence of the node N, so the reverse move uses the same rate. Returns the
     * log of α_mut(n → o) / α_mut(o → n).
     */
    private double resampleBranch(Node node) {
        double startHeight = node.getParent().getHeight();
        double endHeight = node.getHeight();
        double rateScale = this.process.getRateScale(node);
        double jukesCantorRate = this.geneticPrior.branchRateModel.getRateForBranch(node)
                * this.geneticPrior.getTotalMutationRate(node) / this.mutations.getReferenceSequence().length;

        // the sites whose states differ between the parent and the node

        MutationPaths.collectChanges(this.mutations, node.getParent(), node, endHeight, this.branchChanges);
        this.differingSites.clear();
        for (int slot = 0; slot < this.branchChanges.getSize(); slot++) {
            int startState = this.branchChanges.getStartState(slot);
            int endState = this.branchChanges.getEndState(slot);
            if (startState != endState) {
                this.differingSites.addSite(this.branchChanges.getSite(slot), startState, endState);
            }
        }
        int numDifferingSites = this.differingSites.getSize();

        List<Mutation> oldMutations = this.mutations.getMutations(node);
        double logBackwardDensity = this.historySampler.computeLogBranchHistoryDensity(
                oldMutations, startHeight, endHeight, rateScale, jukesCantorRate, numDifferingSites
        );

        // the agreeing sites keep the state of the node, which the move does not change

        List<Mutation> newMutations = this.historySampler.sampleBranchHistory(
                node.getNr(), startHeight, endHeight, rateScale, jukesCantorRate, this.differingSites, this.endStates.resetFor(node)
        );
        double logForwardDensity = this.historySampler.computeLogBranchHistoryDensity(
                newMutations, startHeight, endHeight, rateScale, jukesCantorRate, numDifferingSites
        );

        this.mutations.applyMutations(node, this.mutationLinker.linkMutations(newMutations), this);

        return logBackwardDensity - logForwardDensity;
    }

}
