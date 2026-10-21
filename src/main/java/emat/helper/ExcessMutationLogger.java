package emat.helper;

import beast.base.core.BEASTObject;
import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.core.Loggable;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.TreeInterface;
import emat.state.Mutation;
import emat.state.Mutations;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

@Description("Logs the mutations that exceed the parsimony score of the current tree. For every site with more " +
        "mutations than the Fitch parsimony score of the site, the site, the number of excess mutations and the " +
        "branches carrying its mutations are written as site:excess@node(leaves,height) entries.")
public class ExcessMutationLogger extends BEASTObject implements Loggable {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to log the excess of", Input.Validate.REQUIRED);

    // written if no site has excess mutations, so that the column is never empty
    private static final String NO_EXCESS = "-";

    private Mutations mutations;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
    }

    @Override
    public void init(PrintStream out) {
        out.print(this.getID() + "\t");
    }

    @Override
    public void log(long sample, PrintStream out) {
        out.print(this.formatExcessMutations() + "\t");
    }

    @Override
    public void close(PrintStream out) {
        // nothing to close
    }

    /** Returns the entries of all sites with excess mutations, separated by semicolons. */
    private String formatExcessMutations() {
        TreeInterface tree = this.mutations.getTree();
        int[] parsimonyScores = new FitchParsimony(this.mutations.getAlignment(), tree).getSiteScores();
        List<List<Mutation>> mutationsAtSite = this.collectMutationsBySite(tree, parsimonyScores.length);

        List<String> entries = new ArrayList<>();
        for (int site = 0; site < parsimonyScores.length; site++) {
            int excess = mutationsAtSite.get(site).size() - parsimonyScores[site];
            if (excess > 0) {
                entries.add(this.formatSite(tree, site, excess, mutationsAtSite.get(site)));
            }
        }

        return entries.isEmpty() ? NO_EXCESS : String.join(";", entries);
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

    private String formatSite(TreeInterface tree, int site, int excess, List<Mutation> siteMutations) {
        List<String> branches = new ArrayList<>();
        for (Mutation mutation : siteMutations) {
            Node node = tree.getNode(mutation.nodeNr());
            branches.add(node.getNr() + "(" + node.getLeafNodeCount() + "," + node.getHeight() + ")");
        }

        return site + ":" + excess + "@" + String.join(",", branches);
    }

}
