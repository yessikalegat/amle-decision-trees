# Decision Trees

A decision tree is a flowchart of yes/no questions that the data writes for itself, and it
is the one model you can read from top to bottom. This repository builds trees for both
kinds of task — **predicting a class** and **predicting a number** — chooses a split by hand
before letting scikit-learn choose it, and then spends most of its time on the two things
that decide whether a tree is any use: stopping it from memorizing the training data, and
working out what it actually learned.

Two datasets run through it. **Census records** of 45,222 people, where the task is whether
someone earns more than $50,000, carries the classification half. **1,030 concrete mixes**,
where the task is the strength of the hardened concrete in megapascals, carries the
regression half. Both ship in `data/`, so every notebook runs on its own and offline.

## Learning Objectives

By the end of this repository, you should be able to:

- Build decision tree classifiers and regressors with scikit-learn, and read the questions,
  `samples` and `value` rows of a fitted tree.
- Compute Gini impurity and the gain of a split by hand, and identify the split a greedy
  search will choose.
- Encode categorical columns for a scikit-learn tree, and explain why one-hot encoding is
  preferable to numbering the categories.
- Diagnose overfitting from the gap between training and test scores, and control it with
  `max_depth`, `min_samples_leaf` and `ccp_alpha`.
- Measure which features a fitted model relied on by permuting them in the test data, and
  read where its errors concentrate rather than only its score.
- Tune a tree with cross-validation, score it once on the test set, and state what it
  predicts beyond the range of its training data.

## Learning Path

**Scope** marks the core path every student is expected to complete. Optional lessons stay in
the repository and are worth returning to, but the day does not depend on them. **Units** are
indicative pacing, where one unit is 45 minutes.

| File / Folder | Description | Scope | Units |
| --- | --- | --- | --- |
| [**1 - How a Tree Splits**](01_how_a_tree_splits.ipynb) | Six points you can hold in your head: Gini impurity, the gain of a split scored by hand, the greedy search that scikit-learn runs, and the axis-aligned boundary it draws. | Core | 1.5 |
| [**2 - A Classification Tree**](02_classification_tree.ipynb) | The census data: encoding the categorical columns, a tree that memorizes 36,000 rows, and three ways to stop it — `max_depth`, `min_samples_leaf` and cost-complexity pruning. | Core | 1.5 |
| [**3 - A Regression Tree**](03_regression_tree.ipynb) | The same algorithm predicting a number: variance as the splitting criterion, the staircase it predicts, and the one thing it cannot do — extrapolate beyond its training data. | Core | 1.5 |
| [**4 - Reading a Fitted Tree**](04_reading_a_fitted_tree.ipynb) | `feature_importances_` recomputed by hand, then caught ranking a column of random numbers above a real ingredient, and permutation importance as the honest alternative. | Core | 1.25 |
| [**5 - The Whole Cycle, On Your Own**](05_exercise_full_cycle.ipynb) | Split, baseline, tune with cross-validation, score once, then error analysis and a final importance check. No new material, seven exercises, all of it yours to write. | Core | 2.25 |

### Additional Folders and Files

| File / Folder | Description |
|---|---|
| [**Data**](data/) | `census.csv`: 45,222 census records, 8 categorical and 5 numeric columns, with `income` as the target. `concrete.csv`: 1,030 concrete mixes, seven ingredients and an age in days, with `compressive_strength` in MPa as the target. Both are modified from the UCI originals; see the references below for what changed. |
| [**Solutions**](solutions/) | Reference solutions for every notebook. |
| [**pyproject.toml**](pyproject.toml) | Project configuration and dependencies. |
| [**uv.lock**](uv.lock) | Dependency lock file. |

## Setup

> [!NOTE]
> Throughout these steps, text in angle brackets like `<repo-name>` is a **placeholder**. Replace it, including the `< >` brackets, with your own value. For example, `cd <repo-name>` becomes `cd amle-decision-trees`.

### 1. Create the Repository from the Template

Click **Use this template** on GitHub.

When creating the repository:

- Set yourself as the **Owner**
- Choose a repository name
- Disable **Include all branches**
- Click **Create repository**

> [!IMPORTANT]
> If you are working in pairs or groups, only **one person** should complete this step.

---

### 2. Add Collaborators (Pairs/Groups Only)

If working with teammates:

1. Open the repository on GitHub
2. Go to **Settings → Collaborators**
3. Add your teammates as collaborators
4. Share the repository link with your team

Teammates should accept the invitation before continuing.

---

### 3. Clone the Repository

Copy the SSH URL from the **Code** button on GitHub, then run:

```bash
git clone <copied-ssh-url>
```

The copied SSH URL will look like `git@github.com:<your-username>/<repo-name>.git`.

---

### 4. Move into the Project Folder and Install Dependencies

This installs all dependencies and creates a virtual environment in `.venv/`.

```bash
cd <repo-name>
uv sync
```

---

### 5. Open the Notebook

> [!NOTE]
> Make sure you open VS Code from the project root so it automatically detects the environment created by `uv sync`.

Launch VS Code in the project root folder:

```bash
code .
```

Then open `01_how_a_tree_splits.ipynb`, selecting the Python environment created by `uv sync` as the kernel. Work through the notebooks in order. All five are the core path.

## References & Further Reading

- [**scikit-learn: Decision Trees**](https://scikit-learn.org/stable/modules/tree.html): The user guide behind every model here, including the splitting criteria for classification and regression.
- [**DecisionTreeClassifier**](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) and [**DecisionTreeRegressor**](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html): The full list of hyperparameters, most of which are ways to stop the tree growing.
- [**Minimal cost-complexity pruning**](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning): What `ccp_alpha` does, with the cost function written out.
- [**Permutation feature importance**](https://scikit-learn.org/stable/modules/permutation_importance.html): The measure used in notebooks 04 and 05, and why scikit-learn recommends it over the impurity-based one.
- [**Permutation Importance vs Random Forest Feature Importance**](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance.html): The cardinality trap of notebook 04, demonstrated by scikit-learn itself.
- [**Understanding the decision tree structure**](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html): How to read thresholds and decision paths out of `tree_`, which notebook 04 uses to recompute importances.
- [**UCI: Concrete Compressive Strength**](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength): The regression dataset, collected by I-Cheng Yeh and licensed CC BY 4.0. **Modified here:** converted from the original `.xls` to CSV and the nine columns renamed to short snake_case names. No rows or values were changed.
- [**UCI: Adult**](https://archive.ics.uci.edu/dataset/2/adult): The census dataset, extracted from the 1994 US census by Barry Becker and Ronny Kohavi, and licensed CC BY 4.0. **Modified here:** the 48,842 original rows are reduced to the 45,222 with no missing values, two columns are renamed (`education_level`, `education-num`), and the leading spaces the original carries inside its category values have been stripped.
