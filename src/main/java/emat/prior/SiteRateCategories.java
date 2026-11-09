package emat.prior;

import beast.base.spec.inference.parameter.IntVectorParam;

/**
 * The rate categories of the sites of a genetic prior. Operators change the categories of the state
 * as a whole, and then mark them as redrawn. The genetic prior and the substitution process read a
 * snapshot of the categories, which the genetic prior updates in every calculation. The snapshot is
 * stored with the branches, so that on restore the categories match the cached branches again. The
 * generation counts the changes of the snapshot, so that its readers can tell whether it changed.
 */
public class SiteRateCategories {

    // the categories of the state, which is null if the site model has a single category and no input is given
    private final IntVectorParam<?> parameter;

    // the categories of the snapshot, and of the snapshot stored with the branches
    private int[] siteCategories;
    private int[] storedSiteCategories;

    // whether the state was marked as redrawn since the last update, and since it was stored
    private boolean isChanged = false;
    private boolean isChangedSinceStore = false;

    private long generation = 0;

    public SiteRateCategories(IntVectorParam<?> parameter, int numSites, int numCategories) {
        this.parameter = parameter;
        this.siteCategories = new int[numSites];
        this.storedSiteCategories = new int[numSites];

        if (parameter == null) {
            if (numCategories > 1) {
                throw new IllegalArgumentException("siteRateCategories is required if the site model has more than one category.");
            }
        } else {
            if (parameter.size() != numSites) {
                throw new IllegalArgumentException("siteRateCategories should hold one category per site.");
            }
            for (int site = 0; site < numSites; site++) {
                int category = parameter.get(site);
                if (category < 0 || category >= numCategories) {
                    throw new IllegalArgumentException("siteRateCategories should hold categories from 0 to " + (numCategories - 1) + ".");
                }
            }

            // the snapshot starts with the categories of the state
            this.isChanged = true;
            this.update();
        }
    }

    /** Returns whether the categories are held by the state, which is false if the site model has a single category and no input is given. */
    public boolean hasParameter() {
        return this.parameter != null;
    }

    /**
     * Copies the categories of the state into the snapshot if they were marked as redrawn since the
     * last update, and returns whether they were. The categories are redrawn as a whole, so the
     * snapshot is copied as a whole.
     */
    public boolean update() {
        if (this.parameter == null || !this.isChanged) {
            return false;
        }
        this.isChanged = false;

        for (int site = 0; site < this.siteCategories.length; site++) {
            this.siteCategories[site] = this.parameter.get(site);
        }
        this.generation++;
        return true;
    }

    /** Returns the category of the given site in the snapshot, as of the last update. */
    public int getCategory(int site) {
        return this.siteCategories[site];
    }

    /** Returns the category of the given site in the state, which may differ from the snapshot. */
    public int getStateCategory(int site) {
        return this.parameter.get(site);
    }

    /** Sets the category of the given site in the state. The categories have to be marked as redrawn afterwards. */
    public void setStateCategory(int site, int category) {
        this.parameter.set(site, category);
    }

    /** Marks the categories of the state as redrawn, so that the next update copies them into the snapshot. */
    public void markRedrawn() {
        this.isChanged = true;
        this.isChangedSinceStore = true;
    }

    /** Returns the generation of the snapshot, which changes whenever the snapshot may have changed. */
    public long getGeneration() {
        return this.generation;
    }

    /** Stores the snapshot, so that it can be restored with the branches. */
    public void store() {
        System.arraycopy(this.siteCategories, 0, this.storedSiteCategories, 0, this.siteCategories.length);
        this.isChangedSinceStore = false;
    }

    /**
     * Restores the stored snapshot. The state is restored to its stored categories as well, so the
     * generation changes if the state was redrawn since it was stored.
     */
    public void restore() {
        int[] siteCategories = this.siteCategories;
        this.siteCategories = this.storedSiteCategories;
        this.storedSiteCategories = siteCategories;

        if (this.isChangedSinceStore) {
            this.generation++;
            this.isChangedSinceStore = false;
        }
    }

}
