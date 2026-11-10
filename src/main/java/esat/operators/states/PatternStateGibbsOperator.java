package esat.operators.states;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import esat.prior.GeneticPrior;
import esat.state.NodeStates;
import esat.stochasticmapping.GlobalStateSampler;
import esat.stochasticmapping.SubstitutionProcess;

@Description("Resamples the states of every node at the sites of one pattern, drawn uniformly, from their exact " +
        "conditional distribution given the tree, the model of the genetic prior, the tip data and the states at " +
        "all other sites. As a Gibbs move, it is always accepted. A proposal costs a pruning pass over the tree " +
        "for one pattern, independently of the number of sites with that pattern.")
public class PatternStateGibbsOperator extends Operator {

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
     * Replaces the states at the sites of a uniformly drawn pattern by a draw from their joint
     * conditional distribution given everything else, so the Hastings ratio is infinite. The
     * pattern is drawn independently of the state, so the move is a Gibbs update.
     */
    @Override
    public double proposal() {
        this.substitutionProcess.update();
        int patternNr = Randomizer.nextInt(this.sampler.getPatternCount());
        GlobalStateSampler.PatternSample sample = this.sampler.samplePattern(patternNr);

        this.nodeStates.replaceStatesAtSites(sample.sites(), sample.rootStates(), sample.changesAtNode(), this);
        return Double.POSITIVE_INFINITY;
    }

}
