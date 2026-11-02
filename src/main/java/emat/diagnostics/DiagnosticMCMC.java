package emat.diagnostics;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.util.Randomizer;
import emat.adapters.TreeIntervalAdapter;
import emat.prior.GeneticPrior;
import mcmc.JumpMCMC;
import org.xml.sax.SAXException;
import slice.LinCombSliceOperator;
import slice.SliceOperator;

import javax.xml.parsers.ParserConfigurationException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

@Description("An MCMC chain with the steps of the JumpMCMC that hands every step to a ChainDiagnostics, which " +
        "records it before a rejected proposal is restored. It runs for the full chain length and does not stop at " +
        "a target ESS.")
public class DiagnosticMCMC extends JumpMCMC {

    final public Input<ChainDiagnostics> diagnosticsInput = new Input<>("diagnostics", "the recorder of the steps", Input.Validate.REQUIRED);

    final public Input<List<Operator>> probeOperatorsInput = new Input<>("probeOperator", "operators whose proposals at the probe steps are evaluated and recorded without changing the chain", new ArrayList<>());

    private static final int PROFILE_HALF_WIDTH = 10;
    private static final double PROFILE_STEP = 0.001;

    private ChainDiagnostics diagnostics;
    private boolean isInitialised;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.diagnostics = this.diagnosticsInput.get();
    }

    @Override
    public void run() throws IOException, SAXException, ParserConfigurationException {
        try {
            super.run();
        } finally {
            this.diagnostics.close();
        }
    }

    /**
     * Performs a single MCMC step like the JumpMCMC, and records its outcome, the log posterior
     * of the proposal and the proposed state before a rejected proposal is restored.
     */
    @Override
    protected Operator propagateState(final long sampleNr) {
        if (!this.isInitialised) {
            this.initDiagnostics();
        }

        if (this.diagnostics.isProbeStep(sampleNr)) {
            this.probeOperators(sampleNr);
        }

        long start = System.nanoTime();
        StepAnnotations.clear();
        this.diagnostics.prepareStep(sampleNr);
        this.state.store(sampleNr);

        Operator operator = this.operatorSchedule.selectOperator();

        if (operator instanceof SliceOperator sliceOperator) {
            this.proposeJump(sampleNr, operator, sliceOperator, start);
        } else {
            this.proposeStandard(sampleNr, operator, start);
        }

        this.log(sampleNr);
        return operator;
    }

    /**
     * Draws proposals of every probed operator from the current state and records the log acceptance ratio and
     * the proposed heights of each. The proposals are evaluated like the proposals of the chain but always
     * restored, and the operators are not told about them, so the chain and the tuning of the operators stay as they are.
     */
    private void probeOperators(long sampleNr) {
        for (Operator operator : this.probeOperatorsInput.get()) {
            this.diagnostics.beginProbe(sampleNr, this.operatorSchedule.operators.indexOf(operator));
            for (int i = 0; i < this.diagnostics.probeCountInput.get(); i++) {
                this.probeProposal(sampleNr, operator);
            }
            if (this.diagnostics.isProfiled() && operator instanceof LinCombSliceOperator sliceOperator) {
                this.profileLine(sampleNr, sliceOperator);
            }
        }
        StepAnnotations.clear();
    }

    private void probeProposal(long sampleNr, Operator operator) {
        if (operator instanceof SliceOperator sliceOperator) {
            this.probeSliceProposal(sampleNr, sliceOperator);
            return;
        }

        this.state.store(sampleNr);
        double logHastingsRatio = operator.proposal();

        if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
            this.diagnostics.writeProbe(Double.NEGATIVE_INFINITY, false);
            this.state.restore();
            if (!operator.requiresStateInitialisation()) {
                this.state.setEverythingDirty(false);
                this.state.restoreCalculationNodes();
            }
            return;
        }

        if (operator.requiresStateInitialisation()) {
            this.state.storeCalculationNodes();
            this.state.checkCalculationNodesDirtiness();
        }

        double newLogLikelihood = this.posterior.calculateLogP();
        this.diagnostics.writeProbe(newLogLikelihood - this.oldLogLikelihood + logHastingsRatio, newLogLikelihood != Double.NEGATIVE_INFINITY);

        this.state.restore();
        this.state.restoreCalculationNodes();
        this.state.setEverythingDirty(false);
    }

    /**
     * Evaluates the posterior on a grid of offsets of the line coordinate of the tree interval adapter of the given
     * slice operator and writes the tree prior, the genetic prior and the contributions of the branches at every
     * offset. The state is restored afterwards.
     */
    private void profileLine(long sampleNr, LinCombSliceOperator sliceOperator) {
        TreeIntervalAdapter adapter = sliceOperator.adaptersInput.get().stream()
                .filter(TreeIntervalAdapter.class::isInstance).map(TreeIntervalAdapter.class::cast).findFirst().orElse(null);
        if (adapter == null) {
            return;
        }

        this.state.store(sampleNr);
        adapter.refresh();
        double coordinate = adapter.getMutable(0)[0];
        GeneticPrior geneticPrior = (GeneticPrior) this.diagnostics.geneticPriorInput.get();

        this.diagnostics.beginProfile(sampleNr);
        for (int i = -PROFILE_HALF_WIDTH; i <= PROFILE_HALF_WIDTH; i++) {
            double offset = i * PROFILE_STEP;
            adapter.update(new double[]{coordinate + offset}, 0);
            this.state.checkCalculationNodesDirtiness();
            this.posterior.calculateLogP();
            this.diagnostics.writeProfilePoint(offset, geneticPrior.getBranchLogPs());
        }

        this.state.restore();
        this.recalculateAfterRestore(sliceOperator);
        this.state.setEverythingDirty(false);
    }

    /** Probes a slice operator, which draws a new state from the posterior along a line and is accepted unless it fails. */
    private void probeSliceProposal(long sampleNr, SliceOperator sliceOperator) {
        this.state.store(sampleNr);
        Supplier<Double> computeCurrentLogLikelihood = () -> {
            this.state.checkCalculationNodesDirtiness();
            return this.posterior.calculateLogP();
        };

        double logHastingsRatio = sliceOperator.proposal(computeCurrentLogLikelihood, this.state);

        if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
            this.diagnostics.writeProbe(Double.NEGATIVE_INFINITY, false);
        } else {
            this.state.checkCalculationNodesDirtiness();
            double newLogLikelihood = this.posterior.calculateLogP();
            this.diagnostics.writeProbe(newLogLikelihood - this.oldLogLikelihood + logHastingsRatio, newLogLikelihood != Double.NEGATIVE_INFINITY);
        }

        this.state.restore();
        this.recalculateAfterRestore(sliceOperator);
        this.state.setEverythingDirty(false);
    }

    private void initDiagnostics() {
        try {
            this.diagnostics.writeLegend(this.operatorSchedule.operators);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        this.diagnostics.writeHeader();
        this.diagnostics.initialiseCurrentState();
        this.isInitialised = true;
    }

    /** The standard incremental Metropolis-Hastings step. */
    private void proposeStandard(long sampleNr, Operator operator, long start) {
        double oldLogP = this.oldLogLikelihood;
        double logHastingsRatio = operator.proposal();

        if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
            this.logAlpha = Double.NEGATIVE_INFINITY;
            if (sampleNr >= 0) {
                operator.reject(-2);
                this.record(sampleNr, operator, ChainDiagnostics.OUTCOME_INVALID_PROPOSAL, logHastingsRatio, oldLogP, Double.NaN, start);
            }
            this.state.restore();
            if (!operator.requiresStateInitialisation()) {
                this.state.setEverythingDirty(false);
                this.state.restoreCalculationNodes();
            }
            return;
        }

        if (operator.requiresStateInitialisation()) {
            this.state.storeCalculationNodes();
            this.state.checkCalculationNodesDirtiness();
        }

        this.newLogLikelihood = this.posterior.calculateLogP();
        this.logAlpha = this.newLogLikelihood - this.oldLogLikelihood + logHastingsRatio;

        if (this.isAccepted()) {
            this.oldLogLikelihood = this.newLogLikelihood;
            this.state.acceptCalculationNodes();
            if (sampleNr >= 0) {
                operator.accept();
                this.record(sampleNr, operator, ChainDiagnostics.OUTCOME_ACCEPTED, logHastingsRatio, oldLogP, this.newLogLikelihood, start);
            }
        } else {
            if (sampleNr >= 0) {
                operator.reject(this.newLogLikelihood == Double.NEGATIVE_INFINITY ? -1 : 0);
                this.record(sampleNr, operator, this.rejectionOutcome(), logHastingsRatio, oldLogP, this.newLogLikelihood, start);
            }
            this.state.restore();
            this.state.restoreCalculationNodes();
        }
        this.state.setEverythingDirty(false);
    }

    /** The step of an operator that evaluates the posterior repeatedly inside a proposal, which never trusts the calculation cache to restore. */
    private void proposeJump(long sampleNr, Operator operator, SliceOperator sliceOperator, long start) {
        double oldLogP = this.oldLogLikelihood;
        Supplier<Double> computeCurrentLogLikelihood = () -> {
            this.state.checkCalculationNodesDirtiness();
            return this.posterior.calculateLogP();
        };

        double logHastingsRatio = sliceOperator.proposal(computeCurrentLogLikelihood, this.state);

        if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
            this.logAlpha = Double.NEGATIVE_INFINITY;
            if (sampleNr >= 0) {
                operator.reject(-2);
                this.record(sampleNr, operator, ChainDiagnostics.OUTCOME_INVALID_PROPOSAL, logHastingsRatio, oldLogP, Double.NaN, start);
            }
            this.state.restore();
            this.recalculateAfterRestore(sliceOperator);
            this.state.setEverythingDirty(false);
            return;
        }

        this.state.checkCalculationNodesDirtiness();
        this.newLogLikelihood = this.posterior.calculateLogP();
        this.logAlpha = this.newLogLikelihood - this.oldLogLikelihood + logHastingsRatio;

        if (this.isAccepted()) {
            this.oldLogLikelihood = this.newLogLikelihood;
            this.state.acceptCalculationNodes();
            if (sampleNr >= 0) {
                operator.accept();
                this.record(sampleNr, operator, ChainDiagnostics.OUTCOME_ACCEPTED, logHastingsRatio, oldLogP, this.newLogLikelihood, start);
            }
        } else {
            if (sampleNr >= 0) {
                operator.reject(this.newLogLikelihood == Double.NEGATIVE_INFINITY ? -1 : 0);
                this.record(sampleNr, operator, this.rejectionOutcome(), logHastingsRatio, oldLogP, this.newLogLikelihood, start);
            }
            this.state.restore();
            this.recalculateAfterRestore(sliceOperator);
        }
        this.state.setEverythingDirty(false);
    }

    private boolean isAccepted() {
        return this.logAlpha >= 0 || (this.logAlpha != Double.NEGATIVE_INFINITY && Randomizer.nextDouble() < Math.exp(this.logAlpha));
    }

    private int rejectionOutcome() {
        return this.newLogLikelihood == Double.NEGATIVE_INFINITY ? ChainDiagnostics.OUTCOME_REJECTED_IMPOSSIBLE : ChainDiagnostics.OUTCOME_REJECTED;
    }

    private void record(long sampleNr, Operator operator, int outcome, double logHastingsRatio, double oldLogP, double newLogP, long start) {
        int operatorNr = this.operatorSchedule.operators.indexOf(operator);
        this.diagnostics.recordStep(sampleNr, operatorNr, operator, outcome, this.logAlpha, logHastingsRatio, oldLogP, newLogP, System.nanoTime() - start);
    }

    /** Recalculates the posterior after a rejected jump proposal, whose inner walk corrupts the incremental cache. */
    private void recalculateAfterRestore(SliceOperator sliceOperator) {
        sliceOperator.markDirtyAfterRestore(this.state);
        this.state.checkCalculationNodesDirtiness();
        this.posterior.calculateLogP();
        this.state.setEverythingDirty(false);
        this.state.acceptCalculationNodes();
    }

}
