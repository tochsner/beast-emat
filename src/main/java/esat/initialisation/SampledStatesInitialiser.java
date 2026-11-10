package esat.initialisation;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import esat.stochasticmapping.GlobalStateSampler;
import esat.stochasticmapping.SubstitutionProcess;

import java.util.List;

@Description("Initialises the node states with a draw from their exact conditional distribution given the current " +
        "tree, the model of the genetic prior and the tip data. The initial states are therefore always compatible " +
        "with the data. Initialisers of the tree must come before this one.")
public class SampledStatesInitialiser extends BEASTObject implements StateNodeInitialiser {

    final public Input<NodeStates> nodeStatesInput = new Input<>("nodeStates", "the node states to initialise", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    @Override
    public void initAndValidate() {
    }

    @Override
    public void initStateNodes() {
        NodeStates nodeStates = this.nodeStatesInput.get();

        SubstitutionProcess substitutionProcess = new SubstitutionProcess(this.geneticPriorInput.get());
        substitutionProcess.update();
        GlobalStateSampler.Sample sample = new GlobalStateSampler(nodeStates, substitutionProcess).sample();

        nodeStates.initialiseStates(sample.rootSequence(), List.of(sample.changesAtNode()));
    }

    @Override
    public void getInitialisedStateNodes(List<StateNode> stateNodes) {
        stateNodes.add(this.nodeStatesInput.get());
    }

}
