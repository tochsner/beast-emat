package emat.benchmark;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Loggable;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import emat.helper.MutationPaths;
import emat.helper.SiteChanges;
import emat.prior.GeneticPrior;
import emat.state.Mutations;
import emat.stochasticmapping.SiteHistorySampler;
import emat.stochasticmapping.SubstitutionProcess;
import emat.stochasticmapping.UniformisedStochasticMapping;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Description("Logs the number of sites at which the node states of the current history differ from those of a " +
        "reference history, summed over the root, the other inner nodes and the tips. The reference is an exact draw " +
        "of the node states of all sites given the tree when logging starts, so the tree must stay fixed. As the " +
        "reference is independent of the chain, the distances are stationary and their effective sample sizes tell " +
        "how fast the node states mix.")
public class HistoryDistanceLogger extends BEASTObject implements Loggable {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to log the distance of", Input.Validate.REQUIRED);
    final public Input<GeneticPrior> geneticPriorInput = new Input<>("geneticPrior", "the genetic prior that defines the evolutionary model of the reference", Input.Validate.REQUIRED);

    private Mutations mutations;
    private TreeInterface tree;

    // the reference sequence of the mutations when the reference history was drawn, and per node, the states of the
    // reference history at the sites where they differ from it
    private int[] drawnReferenceSequence;
    private Map<Integer, Integer>[] referenceStates;

    // reusable sequence of a node in the current history
    private SiteChanges sequence;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.mutations.getTree();
        this.sequence = new SiteChanges(this.mutations.getReferenceSequence().length);
    }

    /** Draws the reference history once the tree and the parameters are initialised, and writes the header. */
    @Override
    public void init(PrintStream out) {
        this.drawReferenceHistory();
        out.print("rootDistance\tinnerDistance\ttipDistance\t");
    }

    @SuppressWarnings("unchecked")
    private void drawReferenceHistory() {
        this.drawnReferenceSequence = this.mutations.getReferenceSequence().clone();

        int numStates = this.mutations.getAlignment().getMaxStateCount();
        SubstitutionProcess process = new SubstitutionProcess(this.geneticPriorInput.get(), this.tree, numStates);
        SiteHistorySampler sampler = new SiteHistorySampler(this.mutations, process, new UniformisedStochasticMapping(numStates));
        sampler.updateModel();

        this.referenceStates = new Map[this.tree.getNodeCount()];
        for (int nodeNr = 0; nodeNr < this.referenceStates.length; nodeNr++) {
            this.referenceStates[nodeNr] = new HashMap<>();
        }

        for (int site = 0; site < this.drawnReferenceSequence.length; site++) {
            sampler.sampleSiteHistory(site);
            for (Node node : this.tree.getNodesAsArray()) {
                int state = sampler.getNodeState(node);
                if (state != this.drawnReferenceSequence[site]) {
                    this.referenceStates[node.getNr()].put(site, state);
                }
            }
        }
    }

    @Override
    public void log(long sample, PrintStream out) {
        // the sites at which the current reference sequence differs from the one of the draw
        int[] referenceSequence = this.mutations.getReferenceSequence();
        List<Integer> changedReferenceSites = new ArrayList<>();
        for (int site = 0; site < referenceSequence.length; site++) {
            if (referenceSequence[site] != this.drawnReferenceSequence[site]) {
                changedReferenceSites.add(site);
            }
        }

        long rootDistance = 0;
        long innerDistance = 0;
        long tipDistance = 0;

        for (Node node : this.tree.getNodesAsArray()) {
            int distance = this.computeDistance(node, referenceSequence, changedReferenceSites);
            if (node.isRoot()) {
                rootDistance += distance;
            } else if (node.isLeaf()) {
                tipDistance += distance;
            } else {
                innerDistance += distance;
            }
        }

        out.print(rootDistance + "\t" + innerDistance + "\t" + tipDistance + "\t");
    }

    /**
     * Counts the sites at which the current state of the given node differs from its reference
     * state. Only the sites with a mutation on the path of the node, with a reference state that
     * differs from the drawn reference sequence, or where the reference sequences differ can
     * differ, and each of them is counted once.
     */
    private int computeDistance(Node node, int[] referenceSequence, List<Integer> changedReferenceSites) {
        MutationPaths.collectSequence(this.mutations, node, this.sequence);
        Map<Integer, Integer> nodeReferenceStates = this.referenceStates[node.getNr()];

        Set<Integer> candidateSites = new HashSet<>(nodeReferenceStates.keySet());
        candidateSites.addAll(changedReferenceSites);
        for (int slot = 0; slot < this.sequence.getSize(); slot++) {
            candidateSites.add(this.sequence.getSite(slot));
        }

        int distance = 0;
        for (int site : candidateSites) {
            int slot = this.sequence.getSlot(site);
            int state = slot >= 0 ? this.sequence.getEndState(slot) : referenceSequence[site];
            if (state != nodeReferenceStates.getOrDefault(site, this.drawnReferenceSequence[site])) {
                distance++;
            }
        }
        return distance;
    }

    @Override
    public void close(PrintStream out) {
        // nothing to close
    }

}
