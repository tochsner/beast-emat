package emat.operators.nodetimes.height;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.inference.StateNode;
import beast.base.spec.type.RealScalar;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.List;

@Description("Learns the conditional density of the height of an internal node between its parent and its older " +
        "child from grids of log posterior values that a NodeHeightTrainer collects, and serves the fitted predictor to " +
        "the operators that propose node heights from it. The learning stops for good once the budget of training " +
        "batches is used up, so that the adaptation is finite and the cost of the training is bounded.")
public class NodeHeightLearner extends BEASTObject {

    final public Input<StateNode> popSizeInput = new Input<>("popSize", "the population size of the exponential growth coalescent", Input.Validate.REQUIRED);
    final public Input<StateNode> growthRateInput = new Input<>("growthRate", "the growth rate of the exponential growth coalescent", Input.Validate.REQUIRED);
    final public Input<StateNode> clockRateInput = new Input<>("clockRate", "the rate of the strict clock", Input.Validate.REQUIRED);
    final public Input<StateNode> kappaInput = new Input<>("kappa", "the transition-transversion ratio of the HKY model", Input.Validate.REQUIRED);
    final public Input<Integer> maxBatchesInput = new Input<>("maxBatches", "the number of training batches after which the learning stops for good", 2000);
    final public Input<Integer> firstFitBatchesInput = new Input<>("firstFitBatches", "the number of training batches at which the predictor is fitted for the first time, and which doubles up to the maximum for every further fit", 50);

    private int maxBatches;
    private int nextFitSize;
    private List<NodeHeightPredictor.Batch> batches = new ArrayList<>();
    private final NodeHeightPredictor predictor = new NodeHeightPredictor();
    private boolean isFitted;
    private boolean isTrainingComplete;

    // the values of the model parameters, as of the last call of refreshParameters
    private double popSize;
    private double growthRate;
    private double clockRate;
    private double kappa;

    @Override
    public void initAndValidate() {
        this.maxBatches = this.maxBatchesInput.get();
        this.nextFitSize = Math.min(this.firstFitBatchesInput.get(), this.maxBatches);
    }

    /** Reads the current values of the model parameters, which the contexts created afterwards contain. */
    public void refreshParameters() {
        this.popSize = NodeHeightLearner.readScalar(this.popSizeInput.get());
        this.growthRate = NodeHeightLearner.readScalar(this.growthRateInput.get());
        this.clockRate = NodeHeightLearner.readScalar(this.clockRateInput.get());
        this.kappa = NodeHeightLearner.readScalar(this.kappaInput.get());
    }

    /**
     * Returns the context of the given non-root internal node at the heights and mutations it has now, with the
     * model parameters of the last call of refreshParameters.
     */
    public NodeHeightContext createContext(Node node, Mutations mutations) {
        Node olderChild = NodeHeightMover.getOlderChild(node);
        Node youngerChild = NodeHeightMover.getYoungerChild(node);

        return new NodeHeightContext(
                node.getParent().getHeight(), olderChild.getHeight(), youngerChild.getHeight(),
                mutations.getMutations(node).size(), mutations.getMutations(olderChild).size(), mutations.getMutations(youngerChild).size(),
                this.popSize, this.growthRate, this.clockRate, this.kappa, NodeHeightLearner.countLineages(node.getTree().getNodesAsArray(), node.getParent().getHeight()));
    }

    /** Returns the number of branches that span the given height from below, i.e. the lineages just before a coalescence at that height. */
    private static int countLineages(Node[] nodes, double height) {
        int numLineages = 0;
        for (Node node : nodes) {
            if (!node.isRoot() && node.getHeight() < height && node.getParent().getHeight() >= height) {
                numLineages++;
            }
        }
        return numLineages;
    }

    /**
     * Adds a training batch and fits the predictor whenever the number of batches reaches the next fit size, which
     * doubles from fit to fit. The last fit uses all batches of the budget.
     */
    public void addBatch(NodeHeightPredictor.Batch batch) {
        if (this.isTrainingComplete()) {
            return;
        }

        this.batches.add(batch);
        if (this.batches.size() < this.nextFitSize) {
            return;
        }

        this.predictor.fit(this.batches);
        this.isFitted = true;
        this.isTrainingComplete = this.batches.size() >= this.maxBatches;
        this.nextFitSize = Math.min(2 * this.nextFitSize, this.maxBatches);

        System.out.printf("%s: node height predictor fitted on %d batches, penalty %.3g%n",
                this.getID(), this.batches.size(), this.predictor.getPenalty());

        if (this.isTrainingComplete) {
            // the batches are not needed anymore
            this.batches = new ArrayList<>();
        }
    }

    /** Returns whether the budget of training batches is used up, after which the predictor never changes again. */
    public boolean isTrainingComplete() {
        return this.isTrainingComplete;
    }

    public boolean isFitted() {
        return this.isFitted;
    }

    public NodeHeightPredictor getPredictor() {
        return this.predictor;
    }

    private static double readScalar(StateNode parameter) {
        if (parameter instanceof RealScalar<?> scalar) {
            return scalar.get();
        }
        throw new IllegalArgumentException("The parameter " + parameter.getID() + " is not a real scalar.");
    }

}
