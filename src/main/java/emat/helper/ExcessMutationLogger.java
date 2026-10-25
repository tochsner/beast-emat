package emat.helper;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Loggable;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.datatype.DataType;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import emat.state.Mutation;
import emat.state.Mutations;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

@Description("Logs the mutations that exceed the parsimony score of the current tree, split into two columns. A " +
        "branch is unconstrained at a site if all tips below it are compatible with every state there, e.g. due to " +
        "gaps or missing data. The missing column holds the mutations on unconstrained branches, which parsimony " +
        "never needs. The conflict column holds, for every site where the remaining mutations exceed the Fitch " +
        "parsimony score of the site, the number of excess mutations among them. Both columns list their sites as " +
        "site:excess@node(leaves,height)[old>new:count] entries over the branches carrying the counted mutations, " +
        "followed by the frequency of every state transition among them.")
public class ExcessMutationLogger extends BEASTObject implements Loggable {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to log the excess of", Input.Validate.REQUIRED);

    // written if a column has no entries, so that it is never empty
    private static final String NO_EXCESS = "-";

    private Mutations mutations;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
    }

    @Override
    public void init(PrintStream out) {
        out.print(this.getID() + ".conflict\t" + this.getID() + ".missing\t");
    }

    @Override
    public void log(long sample, PrintStream out) {
        TreeInterface tree = this.mutations.getTree();
        Alignment alignment = this.mutations.getAlignment();
        FitchParsimony parsimony = new FitchParsimony(alignment, tree);
        int[] parsimonyScores = parsimony.getSiteScores();
        boolean[][] unconstrained = this.computeUnconstrainedNodes(parsimony, alignment, tree);
        List<List<Mutation>> mutationsAtSite = this.collectMutationsBySite(tree, parsimonyScores.length);

        List<String> conflictEntries = new ArrayList<>();
        List<String> missingEntries = new ArrayList<>();
        for (int site = 0; site < parsimonyScores.length; site++) {
            int patternNr = alignment.getPatternIndex(site);

            List<Mutation> constrainedMutations = new ArrayList<>();
            List<Mutation> missingMutations = new ArrayList<>();
            for (Mutation mutation : mutationsAtSite.get(site)) {
                (unconstrained[mutation.nodeNr()][patternNr] ? missingMutations : constrainedMutations).add(mutation);
            }

            int conflictExcess = constrainedMutations.size() - parsimonyScores[site];
            if (conflictExcess > 0) {
                conflictEntries.add(this.formatSite(tree, alignment.getDataType(), site, conflictExcess, constrainedMutations));
            }
            if (!missingMutations.isEmpty()) {
                missingEntries.add(this.formatSite(tree, alignment.getDataType(), site, missingMutations.size(), missingMutations));
            }
        }

        out.print(this.formatEntries(conflictEntries) + "\t" + this.formatEntries(missingEntries) + "\t");
    }

    @Override
    public void close(PrintStream out) {
        // nothing to close
    }

    /**
     * Determines per node and pattern whether all tips below the node are compatible with
     * every state. Mutations on the branch above such a node do not change the fit to the data.
     */
    private boolean[][] computeUnconstrainedNodes(FitchParsimony parsimony, Alignment alignment, TreeInterface tree) {
        int numPatterns = alignment.getPatternCount();
        long fullStateSet = (1L << alignment.getDataType().getStateCount()) - 1;
        boolean[][] unconstrained = new boolean[tree.getNodeCount()][numPatterns];

        for (Node node : tree.listNodesPostOrder(null, null)) {
            for (int patternNr = 0; patternNr < numPatterns; patternNr++) {
                if (node.isLeaf()) {
                    unconstrained[node.getNr()][patternNr] = parsimony.getStateSet(node, patternNr) == fullStateSet;
                    continue;
                }

                boolean allChildrenUnconstrained = true;
                for (Node child : node.getChildren()) {
                    allChildrenUnconstrained &= unconstrained[child.getNr()][patternNr];
                }
                unconstrained[node.getNr()][patternNr] = allChildrenUnconstrained;
            }
        }

        return unconstrained;
    }

    /** Returns the mutations of every site, ignoring the mutations above the root. */
    private List<List<Mutation>> collectMutationsBySite(TreeInterface tree, int numSites) {
        List<List<Mutation>> mutationsAtSite = new ArrayList<>();
        for (int site = 0; site < numSites; site++) {
            mutationsAtSite.add(new ArrayList<>(0));
        }

        for (Node node : tree.getNodesAsArray()) {
            if (node.isRoot()) {
                continue;
            }
            for (Mutation mutation : this.mutations.getMutations(node)) {
                mutationsAtSite.get(mutation.site()).add(mutation);
            }
        }

        return mutationsAtSite;
    }

    private String formatSite(TreeInterface tree, DataType dataType, int site, int excess, List<Mutation> siteMutations) {
        List<String> branches = new ArrayList<>();
        for (Mutation mutation : siteMutations) {
            Node node = tree.getNode(mutation.nodeNr());
            branches.add(node.getNr() + "(" + node.getLeafNodeCount() + "," + node.getHeight() + ")");
        }

        return site + ":" + excess + "@" + String.join(",", branches) + "[" + this.formatTransitions(dataType, siteMutations) + "]";
    }

    /** Returns the number of mutations per state transition as old>new:count entries, ordered by old and then new state. */
    private String formatTransitions(DataType dataType, List<Mutation> siteMutations) {
        int stateCount = dataType.getStateCount();
        Map<Integer, Integer> transitionCounts = new TreeMap<>();
        for (Mutation mutation : siteMutations) {
            transitionCounts.merge(mutation.oldState() * stateCount + mutation.newState(), 1, Integer::sum);
        }

        List<String> transitions = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : transitionCounts.entrySet()) {
            String oldState = dataType.getCharacter(entry.getKey() / stateCount);
            String newState = dataType.getCharacter(entry.getKey() % stateCount);
            transitions.add(oldState + ">" + newState + ":" + entry.getValue());
        }

        return String.join(",", transitions);
    }

    /** Returns the given entries separated by semicolons, or the placeholder if there are none. */
    private String formatEntries(List<String> entries) {
        return entries.isEmpty() ? NO_EXCESS : String.join(";", entries);
    }

}
