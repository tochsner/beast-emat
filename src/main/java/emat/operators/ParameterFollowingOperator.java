package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.inference.Operator;
import beast.base.inference.OperatorSchedule;
import beast.base.inference.StateNode;
import beast.base.internal.json.JSONException;
import beast.base.internal.json.JSONObject;
import emat.helper.ParameterFollowing;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.List;

@Description("Wraps an operator that changes the tree and lets parameters of the tree prior follow the change: " +
        "after the wrapped proposal, every followed parameter is shifted by its coefficient times the change of " +
        "the log tree length, and the tree and the parameters are accepted or rejected together. The proposal of " +
        "the wrapped operator must not depend on the followed parameters " +
        "(docs/parameter-following-operator.md).")
public class ParameterFollowingOperator extends Operator {

    final public Input<Operator> operatorInput = new Input<>("operator", "the operator whose changes of the tree the parameters follow", Input.Validate.REQUIRED);
    final public Input<ParameterFollowing> followingInput = new Input<>("following", "the followed parameters and their coefficients, shared by all operators that let them follow", Input.Validate.REQUIRED);

    private Operator operator;
    private ParameterFollowing following;

    @Override
    public void initAndValidate() {
        this.operator = this.operatorInput.get();
        this.following = this.followingInput.get();

        // the draws of the batch operator depend on the parameters of its tree prior approximation

        if (this.operator instanceof BatchNodeDisplacementOperator batchOperator) {
            for (StateNode parameter : batchOperator.treePriorParametersInput.get()) {
                if (this.following.getParameters().contains(parameter)) {
                    throw new IllegalArgumentException("The parameter " + parameter.getID() + " cannot follow the tree in " + this.getID() +
                            ", because the proposal of " + this.operator.getID() + " depends on it as a parameter of the tree prior.");
                }
            }
        }
    }

    /**
     * Runs the wrapped proposal and shifts the followed parameters by their coefficients times the change of
     * the summary of the tree. For fixed coefficients, the shift only depends on the old and the new tree and
     * the reverse move undoes it, so the Hastings ratio of the wrapped operator only gains the Jacobian of the
     * shift.
     */
    @Override
    public double proposal() {
        double oldSummary = this.following.computeSummary();
        this.following.observe(oldSummary);

        double logHastingsRatio = this.operator.proposal();
        if (logHastingsRatio == Double.NEGATIVE_INFINITY) {
            return Double.NEGATIVE_INFINITY;
        }

        double summaryChange = this.following.computeSummary() - oldSummary;
        if (summaryChange == 0.0) {
            return logHastingsRatio;
        }

        return logHastingsRatio + this.following.shiftParameters(summaryChange);
    }

    @Override
    public List<StateNode> listStateNodes() {
        List<StateNode> stateNodes = new ArrayList<>(this.operator.listStateNodes());
        for (StateNode parameter : this.following.getParameters()) {
            if (parameter.isEstimatedInput.get() && !stateNodes.contains(parameter)) {
                stateNodes.add(parameter);
            }
        }
        return stateNodes;
    }

    /* Forwarding to the Wrapped Operator */

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

    /* State File */

    @Override
    public void storeToFile(PrintWriter out) {
        StringWriter ownState = new StringWriter();
        PrintWriter ownWriter = new PrintWriter(ownState);
        super.storeToFile(ownWriter);
        ownWriter.flush();

        StringWriter operatorState = new StringWriter();
        PrintWriter operatorWriter = new PrintWriter(operatorState);
        this.operator.storeToFile(operatorWriter);
        operatorWriter.flush();

        try {
            JSONObject json = new JSONObject(ownState.toString());
            json.put("operator", new JSONObject(operatorState.toString()));
            json.put("following", this.following.storeToJson());
            out.print(json.toString());
        } catch (JSONException exception) {
            // the run goes on with the state of this operator alone
            exception.printStackTrace();
            out.print(ownState);
        }
    }

    @Override
    public void restoreFromFile(JSONObject json) {
        super.restoreFromFile(json);

        try {
            if (json.has("operator")) {
                this.operator.restoreFromFile(json.getJSONObject("operator"));
            }
            if (json.has("following")) {
                this.following.restoreFromJson(json.getJSONObject("following"));
            }
        } catch (JSONException exception) {
            // the run goes on with what has been restored, and learns the rest again
            exception.printStackTrace();
        }
    }

}
