package emat.benchmark;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.inference.MCMC;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.operators.MutationDirectedSprOperator;
import org.xml.sax.SAXException;

import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

@Description("An MCMC chain that measures, per operator, the time spent on the proposal, on the subsequent " +
        "posterior evaluation and on accepting or restoring the state. It also counts the accepted proposals " +
        "and, if a tree is given, the accepted proposals that changed its topology. The measurements are written " +
        "to a CSV file after the run.")
public class TimedMCMC extends MCMC {

    final public Input<String> timingFileInput = new Input<>("timingFile", "the CSV file the measurements are written to", Input.Validate.REQUIRED);
    final public Input<Long> timingBurninInput = new Input<>("timingBurnin", "the number of initial steps that are not measured", 0L);
    final public Input<Tree> treeInput = new Input<>("tree", "the tree whose topology changes are counted");

    long timingBurnin;
    Tree tree;

    // per operator, in the order of the operator schedule
    Map<Operator, Integer> operatorIndices;
    long[] numProposals;
    long[] numAccepted;
    long[] numDirectRejects;
    long[] numTopologyChanges;
    long[] proposalNanos;
    long[] evaluationNanos;
    long[] resolutionNanos;

    // over all measured steps
    long numMeasuredSteps;
    long storeNanos;
    long loggingNanos;
    long measurementStart;
    long totalNanos;

    // the parent of every node before the current proposal
    int[] oldParentNrs;

    @Override
    public void initAndValidate() {
        super.initAndValidate();

        this.timingBurnin = this.timingBurninInput.get();
        this.tree = this.treeInput.get();
        if (this.tree != null) {
            this.oldParentNrs = new int[this.tree.getNodeCount()];
        }
    }

    @Override
    public void run() throws IOException, SAXException, ParserConfigurationException {
        super.run();
        this.writeTimings();
    }

    /**
     * Performs a single MCMC step like the base class, and measures its parts once the
     * timing burn-in has passed.
     */
    @Override
    protected Operator propagateState(final long sampleNr) {
        if (this.operatorIndices == null) {
            this.initMeasurements();
        }

        boolean isMeasured = sampleNr >= this.timingBurnin;
        if (isMeasured && this.numMeasuredSteps == 0) {
            this.measurementStart = System.nanoTime();
        }

        long start = System.nanoTime();
        this.state.store(sampleNr);
        final Operator operator = this.operatorSchedule.selectOperator();
        long afterStore = System.nanoTime();

        if (isMeasured && this.tree != null) {
            this.collectParentNrs();
        }

        long beforeProposal = System.nanoTime();
        final double logHastingsRatio = operator.proposal();
        long afterProposal = System.nanoTime();

        long afterEvaluation = afterProposal;
        boolean isAccepted = false;

        if (logHastingsRatio != Double.NEGATIVE_INFINITY) {
            if (operator.requiresStateInitialisation()) {
                this.state.storeCalculationNodes();
                this.state.checkCalculationNodesDirtiness();
            }

            this.newLogLikelihood = this.posterior.calculateLogP();
            afterEvaluation = System.nanoTime();

            this.logAlpha = this.newLogLikelihood - this.oldLogLikelihood + logHastingsRatio;

            if (this.logAlpha >= 0 || (this.logAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(this.logAlpha))) {
                this.oldLogLikelihood = this.newLogLikelihood;
                this.state.acceptCalculationNodes();

                if (sampleNr >= 0) {
                    operator.accept();
                }
                isAccepted = true;
            } else {
                if (sampleNr >= 0) {
                    operator.reject(this.newLogLikelihood == Double.NEGATIVE_INFINITY ? -1 : 0);
                }
                this.state.restore();
                this.state.restoreCalculationNodes();
            }
            this.state.setEverythingDirty(false);
        } else {
            this.logAlpha = Double.NEGATIVE_INFINITY;

            if (sampleNr >= 0) {
                operator.reject(-2);
            }
            this.state.restore();
            if (!operator.requiresStateInitialisation()) {
                this.state.setEverythingDirty(false);
                this.state.restoreCalculationNodes();
            }
        }
        long afterResolution = System.nanoTime();

        this.log(sampleNr);
        long afterLogging = System.nanoTime();

        if (isMeasured) {
            int operatorNr = this.operatorIndices.get(operator);

            this.numProposals[operatorNr]++;
            this.proposalNanos[operatorNr] += afterProposal - beforeProposal;
            this.evaluationNanos[operatorNr] += afterEvaluation - afterProposal;
            this.resolutionNanos[operatorNr] += afterResolution - afterEvaluation;

            if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
                this.numDirectRejects[operatorNr]++;
            }
            if (isAccepted) {
                this.numAccepted[operatorNr]++;
                if (this.tree != null && this.hasTopologyChanged()) {
                    this.numTopologyChanges[operatorNr]++;
                }
            }

            this.numMeasuredSteps++;
            this.storeNanos += afterStore - start;
            this.loggingNanos += afterLogging - afterResolution;
            this.totalNanos = System.nanoTime() - this.measurementStart;
        }

        return operator;
    }

    /* Measurements */

    private void initMeasurements() {
        List<Operator> operators = this.operatorSchedule.operators;
        int numOperators = operators.size();

        this.operatorIndices = new IdentityHashMap<>();
        for (int i = 0; i < numOperators; i++) {
            this.operatorIndices.put(operators.get(i), i);
        }

        this.numProposals = new long[numOperators];
        this.numAccepted = new long[numOperators];
        this.numDirectRejects = new long[numOperators];
        this.numTopologyChanges = new long[numOperators];
        this.proposalNanos = new long[numOperators];
        this.evaluationNanos = new long[numOperators];
        this.resolutionNanos = new long[numOperators];
    }

    private void collectParentNrs() {
        Node[] nodes = this.tree.getNodesAsArray();
        for (int i = 0; i < this.oldParentNrs.length; i++) {
            Node parent = nodes[i].getParent();
            this.oldParentNrs[i] = parent == null ? -1 : parent.getNr();
        }
    }

    /** Checks whether any node has another parent than before the current proposal. */
    private boolean hasTopologyChanged() {
        Node[] nodes = this.tree.getNodesAsArray();
        for (int i = 0; i < this.oldParentNrs.length; i++) {
            Node parent = nodes[i].getParent();
            if (this.oldParentNrs[i] != (parent == null ? -1 : parent.getNr())) {
                return true;
            }
        }
        return false;
    }

    /**
     * Writes the totals over all measured steps as comment lines, followed by one row per
     * operator with its counts and its total nanoseconds per part of the step.
     */
    private void writeTimings() throws IOException {
        List<Operator> operators = this.operatorSchedule.operators;

        try (PrintStream out = new PrintStream(this.timingFileInput.get())) {
            out.println("# steps=" + this.numMeasuredSteps);
            out.println("# total_ns=" + this.totalNanos);
            out.println("# store_ns=" + this.storeNanos);
            out.println("# logging_ns=" + this.loggingNanos);
            for (Operator operator : operators) {
                if (operator instanceof MutationDirectedSprOperator mdSprOperator) {
                    out.println("# mdspr_local_regions=" + mdSprOperator.getMeanNumLocalRegions());
                }
            }
            out.println("operator,spec,weight,proposals,accepted,direct_rejects,topology_changes,proposal_ns,evaluation_ns,resolution_ns");

            for (int i = 0; i < operators.size(); i++) {
                Operator operator = operators.get(i);
                out.println(String.join(",",
                        operator.getID(),
                        operator.getClass().getSimpleName(),
                        Double.toString(operator.getWeight()),
                        Long.toString(this.numProposals[i]),
                        Long.toString(this.numAccepted[i]),
                        Long.toString(this.numDirectRejects[i]),
                        Long.toString(this.numTopologyChanges[i]),
                        Long.toString(this.proposalNanos[i]),
                        Long.toString(this.evaluationNanos[i]),
                        Long.toString(this.resolutionNanos[i])
                ));
            }
        }
    }

}
