package emat.operators;

import beast.base.core.Description;
import beast.base.core.Input;
import beast.base.evolution.alignment.Alignment;
import beast.base.evolution.operator.TreeOperator;
import beast.base.evolution.tree.Node;
import beast.base.evolution.tree.Tree;
import beast.base.util.Randomizer;
import emat.state.Mutation;
import emat.state.Mutations;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@Description("Permutes the tips within a random group of tips with identical data. Each tip takes over the " +
        "position of another tip in the group, i.e. its parent and the mutations on the branch above it. The " +
        "permutation is drawn uniformly among all permutations that keep every tip younger than its new parent " +
        "and than the mutations on its new branch. The node heights and the mutation histories stay the same, " +
        "so this changes the topology without touching the tree prior.")
public class IdenticalTipPermutationOperator extends TreeOperator {

    final public Input<Mutations> mutationsInput = new Input<>("mutations", "the mutations to operate on", Input.Validate.REQUIRED);

    Mutations mutations;
    Tree tree;

    // the node numbers of the tips in each group of at least two tips with identical data
    List<int[]> groups;

    @Override
    public void initAndValidate() {
        this.mutations = this.mutationsInput.get();
        this.tree = this.treeInput.get();
        this.groups = this.collectIdenticalTipGroups();
    }

    /**
     * Groups the tips by their data at all sites, including missing and ambiguous codes, and
     * keeps the groups with at least two tips. The tips keep their numbers and taxa when the
     * tree changes, so the groups stay fixed.
     */
    private List<int[]> collectIdenticalTipGroups() {
        Alignment alignment = this.mutations.getAlignment();
        int numPatterns = alignment.getPatternCount();

        Map<List<Integer>, List<Integer>> tipsByData = new LinkedHashMap<>();
        for (Node tip : this.tree.getExternalNodes()) {
            int taxonIndex = alignment.getTaxonIndex(tip.getID());

            List<Integer> data = new ArrayList<>(numPatterns);
            for (int patternIndex = 0; patternIndex < numPatterns; patternIndex++) {
                data.add(alignment.getPattern(taxonIndex, patternIndex));
            }
            tipsByData.computeIfAbsent(data, key -> new ArrayList<>()).add(tip.getNr());
        }

        List<int[]> groups = new ArrayList<>();
        for (List<Integer> tipNrs : tipsByData.values()) {
            if (tipNrs.size() >= 2) {
                groups.add(tipNrs.stream().mapToInt(Integer::intValue).toArray());
            }
        }
        return groups;
    }

    /**
     * Picks a group uniformly and draws a uniform permutation among the valid ones other than
     * the identity, by redrawing until it is not the identity. The positions of a group, i.e.
     * the parents with the mutations on the branches below them, and the tip heights do not
     * change under the move, so the set of valid permutations is the same before and after
     * it. The proposal is therefore symmetric. If the identity is the only valid permutation,
     * the proposal is rejected. Otherwise, there are at least two valid permutations, so at
     * most half of the draws are the identity.
     */
    @Override
    public double proposal() {
        if (this.groups.isEmpty()) {
            return Double.NEGATIVE_INFINITY;
        }

        int[] group = this.groups.get(Randomizer.nextInt(this.groups.size()));
        Node[] tips = Arrays.stream(group).mapToObj(this.tree::getNode).toArray(Node[]::new);

        int[] newPositions;
        do {
            newPositions = this.drawValidPermutation(tips);
            if (newPositions == null) {
                return Double.NEGATIVE_INFINITY;
            }
        } while (this.isIdentity(newPositions));

        this.applyPermutation(tips, newPositions);
        return 0.0;
    }

    /**
     * Draws a permutation uniformly among those that move every tip to a position whose
     * parent is older than the tip and whose mutations are not younger than it. Returns the
     * index of the new position of every tip, where position j is the current position of
     * tip j, or null if the identity is the only valid permutation. A position that is valid
     * for a tip is also valid for every younger tip. Hence, if the tips are assigned from the
     * oldest to the youngest, the number of free valid positions of each tip does not depend
     * on the earlier choices, and choosing uniformly among them yields a uniform permutation.
     * The number of valid permutations is the product of these numbers.
     */
    private int[] drawValidPermutation(Node[] tips) {
        int numTips = tips.length;

        double[] positionFloors = new double[numTips];
        for (int j = 0; j < numTips; j++) {
            positionFloors[j] = this.getPositionFloor(tips[j]);
        }

        Integer[] tipOrder = new Integer[numTips];
        for (int i = 0; i < numTips; i++) {
            tipOrder[i] = i;
        }
        Arrays.sort(tipOrder, Comparator.comparingDouble((Integer i) -> tips[i].getHeight()).reversed());

        int[] newPositions = new int[numTips];
        boolean[] isTaken = new boolean[numTips];
        int[] candidates = new int[numTips];
        boolean hasAlternative = false;

        for (int i : tipOrder) {
            int numCandidates = 0;
            for (int j = 0; j < numTips; j++) {
                if (!isTaken[j] && this.isValidPosition(tips[i], tips[j], positionFloors[j])) {
                    candidates[numCandidates++] = j;
                }
            }

            // the current position of every tip is valid, so every tip has a candidate
            int position = candidates[Randomizer.nextInt(numCandidates)];
            newPositions[i] = position;
            isTaken[position] = true;
            hasAlternative |= numCandidates > 1;
        }

        return hasAlternative ? newPositions : null;
    }

    /**
     * Returns the height above which every event on the branch of the given position lies:
     * the time of its youngest mutation if there is one, and otherwise the height of its
     * parent.
     */
    private double getPositionFloor(Node positionTip) {
        List<Mutation> branchMutations = this.mutations.getMutations(positionTip);

        // the mutations are sorted by descending height, so the last one is the youngest
        return branchMutations.isEmpty()
                ? positionTip.getParent().getHeight()
                : branchMutations.get(branchMutations.size() - 1).time();
    }

    /** Checks whether the given tip can take over the position of the given position tip. */
    private boolean isValidPosition(Node tip, Node positionTip, double positionFloor) {
        return positionTip.getParent().getHeight() > tip.getHeight() && positionFloor >= tip.getHeight();
    }

    /** Checks whether the given permutation leaves every tip at its current position. */
    private boolean isIdentity(int[] newPositions) {
        for (int i = 0; i < newPositions.length; i++) {
            if (newPositions[i] != i) {
                return false;
            }
        }
        return true;
    }

    /**
     * Moves every tip to its new position: below the parent of the position tip, with the
     * mutations of the branch above the position tip. All tips are detached first, so that
     * cycles of the permutation are handled without intermediate states.
     */
    private void applyPermutation(Node[] tips, int[] newPositions) {
        int numTips = tips.length;

        Node[] oldParents = new Node[numTips];
        List<List<Mutation>> oldBranchMutations = new ArrayList<>(numTips);
        for (Node tip : tips) {
            oldBranchMutations.add(this.mutations.getMutations(tip));
        }

        // detach all tips, then attach each to the parent of its new position

        for (int i = 0; i < numTips; i++) {
            oldParents[i] = tips[i].getParent();
            oldParents[i].removeChild(tips[i]);
        }

        for (int i = 0; i < numTips; i++) {
            Node newParent = oldParents[newPositions[i]];
            newParent.addChild(tips[i]);
            newParent.makeDirty(Tree.IS_FILTHY);
            tips[i].makeDirty(Tree.IS_FILTHY);
        }

        // move the mutations along with the positions, relabelled to the branches of their new tips

        for (int i = 0; i < numTips; i++) {
            int tipNr = tips[i].getNr();

            List<Mutation> newBranchMutations = new ArrayList<>();
            for (Mutation mutation : oldBranchMutations.get(newPositions[i])) {
                newBranchMutations.add(new Mutation(
                        tipNr, mutation.time(), mutation.timeOfPreviousMutation(),
                        mutation.site(), mutation.oldState(), mutation.newState()
                ));
            }
            this.mutations.applyMutations(tips[i], newBranchMutations, this);
        }
    }

}
