package emat.operators.treeprior;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.OperatorSchedule;
import beast.base.inference.StateNode;

import java.util.ArrayList;
import java.util.List;

@Description("Wraps an operator and redraws the parameters of the tree prior in the same proposal, from the distribution " +
        "that is learned given the statistics of the tree that the wrapped operator produced. The wrapped operator is " +
        "accepted or rejected together with the new parameters, with the learned densities of the old and the new " +
        "parameters in the Hastings ratio, so that a change of the tree is not held back by parameters that fit the old " +
        "tree. Until the distribution is fitted, the wrapped operator proposes on its own.")
public class TreePriorResamplingOperator extends LearnedTreePriorOperator {

    final public Input<Operator> operatorInput = new Input<>("operator", "the operator whose proposals come with new values of the parameters of the tree prior", Input.Validate.REQUIRED);

    private Operator operator;

    @Override
    public void initAndValidate() {
        super.initAndValidate();
        this.operator = this.operatorInput.get();
    }

    /**
     * Learns from a sweep if one is due, then proposes with the wrapped operator and redraws the parameters given the new
     * tree. Returns the log Hastings ratio of the wrapped operator, plus the change of the log Jacobian corrections of the
     * adapters, plus the log density of the old parameters given the old tree minus the log density of the new parameters
     * given the new tree.
     */
    @Override
    public double proposal() {
        this.numProposals++;

        double[] oldImmutable = this.readImmutable();
        double[] oldMutable = this.readMutable();
        if (this.numProposals > this.burnInInput.get()) {
            this.sampler.record(this.tree, oldImmutable, oldMutable, this::evaluateLogTarget);
        }
        if (!this.sampler.isFitted()) {
            return this.operator.proposal();
        }

        double[] oldConditions = this.sampler.createConditions(this.tree, oldImmutable);
        double logDensityReverse = this.sampler.computeLogDensity(oldConditions, oldMutable);
        double logCorrectionBefore = this.computeLogJacobianCorrection();

        double logHastingsRatio = this.operator.proposal();
        if (!Double.isFinite(logHastingsRatio)) {
            return Double.NEGATIVE_INFINITY;
        }

        double[] newConditions = this.sampler.createConditions(this.tree, this.readImmutable());
        double[] newMutable = this.sampler.sample(newConditions);
        if (!this.writeMutable(newMutable)) {
            return Double.NEGATIVE_INFINITY;
        }
        double logCorrectionAfter = this.computeLogJacobianCorrection();
        double logDensityForward = this.sampler.computeLogDensity(newConditions, newMutable);

        return logHastingsRatio + logCorrectionBefore - logCorrectionAfter + logDensityReverse - logDensityForward;
    }

    /* The wrapped operator is tuned by the acceptance of the combined proposal. */

    @Override
    public void setOperatorSchedule(OperatorSchedule operatorSchedule) {
        super.setOperatorSchedule(operatorSchedule);
        this.operator.setOperatorSchedule(operatorSchedule);
    }

    @Override
    public void accept() {
        super.accept();
        this.operator.accept();
    }

    @Override
    public void reject(int reason) {
        super.reject(reason);
        this.operator.reject(reason);
    }

    @Override
    public void optimize(double logAlpha) {
        this.operator.optimize(logAlpha);
    }

    @Override
    public double getTargetAcceptanceProbability() {
        return this.operator.getTargetAcceptanceProbability();
    }

    @Override
    public double getCoercableParameterValue() {
        return this.operator.getCoercableParameterValue();
    }

    @Override
    public void setCoercableParameterValue(double value) {
        this.operator.setCoercableParameterValue(value);
    }

    @Override
    public String getPerformanceSuggestion() {
        return this.operator.getPerformanceSuggestion();
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>(this.operator.listStateNodes());
        for (StateNode stateNode : super.listStateNodes()) {
            if (!stateNodes.contains(stateNode)) {
                stateNodes.add(stateNode);
            }
        }
        return stateNodes;
    }

}
