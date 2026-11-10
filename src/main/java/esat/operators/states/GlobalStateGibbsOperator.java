package esat.operators.states;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import esat.stochasticmapping.GlobalStateSampler;
import esat.stochasticmapping.SubstitutionProcess;

@Description("Resamples the states of every node at every site from their exact conditional distribution given the " +
        "tree, the model of the genetic prior and the tip data. As a Gibbs move, it is always accepted. A proposal " +
        "costs about as much as a full evaluation of the tree likelihood.")
public class GlobalStateGibbsOperator extends Operator {

    final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "the node states to operate on", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    NodeStates nodeStates;
    SubstitutionProcess substitutionProcess;
    GlobalStateSampler sampler;

    @Override
    public void initAndValidate() {
        this.nodeStates = this.nodeStatesInput.get();
        this.substitutionProcess = new SubstitutionProcess(this.geneticPriorInput.get());
        this.sampler = new GlobalStateSampler(this.nodeStates, this.substitutionProcess);
    }

    /**
     * Replaces all states by a draw from their joint conditional distribution given
     * everything else, so the Hastings ratio is infinite.
     */
    @Override
    public double proposal() {
        this.substitutionProcess.update();
        GlobalStateSampler.Sample sample = this.sampler.sample();

        this.nodeStates.replaceStates(sample.rootSequence(), sample.changesAtNode(), this);
        return Double.POSITIVE_INFINITY;
    }

}
