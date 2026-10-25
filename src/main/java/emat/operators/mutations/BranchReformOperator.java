package emat.operators.mutations;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.operators.topology.BranchMutations;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.HistorySampler;
import emat.stochasticmapping.SubstitutionProcess;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Description("Reforms the history of a random branch that lies below the root or carries a mutation, without " +
        "changing the tree (branch reform, " +
        "docs/mcmc-moves.md §9.2). On a branch below an internal node, all mutation times are redrawn " +
        "uniformly on the branch, keeping the order of the mutations at each site. On a branch below the " +
        "root, the history of the whole path through the root to the other child is resampled by the chosen " +
        "history sampler, which lets mutations move between the two root branches and changes the root " +
        "sequence. Missations are not supported.")
public class BranchReformOperator extends Operator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);
    final public Input<HistorySampler> historySamplerInput = new Input<>("historySampler", "how the history of the path between the root children is resampled: JUKES_CANTOR under an approximating Jukes-Cantor model, " +
            "HYBRID by choosing the sites with mutations under Jukes-Cantor and then sampling their histories exactly under the model of the genetic prior", HistorySampler.JUKES_CANTOR, HistorySampler.values());

    Mutations mutations;
    TreeInterface tree;
    BranchMutations mutationLinker;
    SubstitutionProcess process;
    RootPathResampler rootPathResampler;

    // the number of uniform draws of a node before the nodes whose branch can be reformed are listed
    private static final int MAX_SAMPLING_ATTEMPTS = 64;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();

        GeneticPrior geneticPrior = this.geneticPriorInput.get();
        int numSites = this.mutations.getReferenceSequence().length;
        this.mutationLinker = new BranchMutations(numSites);
        this.process = new SubstitutionProcess(geneticPrior, this.tree, this.mutations.getAlignment().getMaxStateCount());
        this.rootPathResampler = new RootPathResampler(
                this.mutations, geneticPrior, this.process, this.historySamplerInput.get().create(this.process, numSites)
        );
    }

    /**
     * Picks a branch uniformly among the branches whose history can be reformed, and reforms it. These are the
     * branches below the root and the other branches that carry a mutation. The move keeps the number of
     * mutations on a branch below an internal node and only moves mutations between the two branches below the
     * root, so the set of these branches does not change and the choice is symmetric.
     */
    @Override
    public double proposal() {
        Node node = this.sampleReformableBranch();
        if (node == null) {
            return Double.NEGATIVE_INFINITY;
        }

        if (node.getParent().isRoot()) {
            // both branches below the root reform the same path, which keeps the root height
            this.process.update();
            return this.rootPathResampler.resample(node.getParent(), node.getParent().getHeight(), this);
        } else {
            return this.reformBranch(node);
        }
    }

    /**
     * Returns a node drawn uniformly among the nodes whose branch can be reformed, or null if there is none.
     * Nodes are drawn uniformly until one of them qualifies, which is a uniform draw among the nodes that do.
     */
    private Node sampleReformableBranch() {
        int numNodes = this.tree.getNodeCount();

        for (int attempt = 0; attempt < MAX_SAMPLING_ATTEMPTS; attempt++) {
            Node node = this.tree.getNode(Randomizer.nextInt(numNodes));
            if (this.isReformable(node)) {
                return node;
            }
        }

        // such nodes are rare or missing, so they are listed

        List<Node> reformableNodes = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
            Node node = this.tree.getNode(nodeNr);
            if (this.isReformable(node)) {
                reformableNodes.add(node);
            }
        }

        if (reformableNodes.isEmpty()) {
            return null;
        }
        return reformableNodes.get(Randomizer.nextInt(reformableNodes.size()));
    }

    /**
     * Returns whether the history of the branch above the given node can be reformed: the branch lies below a
     * binary root, or it carries a mutation whose time can be redrawn.
     */
    private boolean isReformable(Node node) {
        if (node.isRoot()) {
            return false;
        }

        Node parent = node.getParent();
        if (parent.isRoot()) {
            return parent.getChildCount() == 2;
        }
        return !this.mutations.getMutations(node).isEmpty();
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
                    mutation.nodeNr(), newTimes[i], Mutation.BRANCH_START,
                    mutation.site(), mutation.oldState(), mutation.newState()
            ));
        }

        this.mutations.applyMutations(node, this.mutationLinker.linkMutations(newBranchMutations), this);

        return 0.0;
    }

}
