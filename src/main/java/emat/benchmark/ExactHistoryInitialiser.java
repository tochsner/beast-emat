package emat.benchmark;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.StateNode;
import beast.base.inference.StateNodeInitialiser;
import emat.operators.topology.BranchMutations;
import emat.prior.GeneticPrior;
import emat.state.Mutation;
import emat.state.Mutations;
import emat.stochasticmapping.SiteHistorySampler;
import emat.stochasticmapping.SubstitutionProcess;
import emat.stochasticmapping.UniformisedStochasticMapping;

import java.util.ArrayList;
import java.util.List;

@Description("Initialises the mutations by an exact draw of the history of every site given the tree and the model " +
        "of the genetic prior, so that a chain on a fixed tree starts in its stationary distribution. It must run after " +
        "the tree and the parameters are initialised.")
public class ExactHistoryInitialiser extends BEASTObject implements StateNodeInitialiser {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to initialise", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model", Input.Validate.REQUIRED);

    private Mutations mutations;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
    }

    @Override
    public void initStateNodes() {
        int numNodes = this.mutations.getTree().getNodeCount();
        int numSites = this.mutations.getReferenceSequence().length;
        int numStates = this.mutations.getAlignment().getMaxStateCount();

        SubstitutionProcess process = new SubstitutionProcess(this.geneticPriorInput.get(), this.mutations.getTree(), numStates);
        SiteHistorySampler sampler = new SiteHistorySampler(this.mutations, process, new UniformisedStochasticMapping(numStates));
        sampler.updateModel();

        List<List<Mutation>> history = new ArrayList<>();
        for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
            history.add(new ArrayList<>());
        }
        for (int site = 0; site < numSites; site++) {
            List<List<Mutation>> siteHistory = sampler.sampleSiteHistory(site);
            for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
                history.get(nodeNr).addAll(siteHistory.get(nodeNr));
            }
        }

        BranchMutations mutationLinker = new BranchMutations(numSites);
        for (int nodeNr = 0; nodeNr < numNodes; nodeNr++) {
            history.set(nodeNr, mutationLinker.linkMutations(history.get(nodeNr)));
        }
        this.mutations.initialiseMutations(history);
    }

    @Override
    public void getInitialisedStateNodes(List<StateNode> stateNodes) {
        stateNodes.add(this.mutations);
    }

}
