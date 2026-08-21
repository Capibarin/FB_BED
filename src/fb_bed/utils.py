import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import anndata as ad
import pandas as pd
import scanpy as sc
    
def plot_simulation(generated_samples, data_samples, condition, size=1, plot_size=(6, 6)):
    """
    Visualize generated samples against true data samples and display
    condition-based coloring.

    This function produces three plots:
    1. Overlay of generated and true samples.
    2. Generated samples colored by condition.
    3. True data samples colored by condition.

    Args:
        generated_samples (np.ndarray): Array of generated samples with shape (n_samples, 2).
        data_samples (np.ndarray): Array of true data samples with shape (n_samples, 2).
        condition (np.ndarray): Condition matrix of shape (n_samples, n_conditions).
        size (float, optional): Marker size for scatter plots. Defaults to 1.
        plot_size (tuple, optional): Figure size (width, height). Defaults to (6, 6).

    Returns:
        None
    """
    # Assign numeric color labels based on unique condition rows
    color = np.zeros(condition.shape[0])
    unique_conditions = np.unique(condition, axis=0)

    for i in range(unique_conditions.shape[0]):
        mask = np.all(condition == unique_conditions[i], axis=1)
        color[mask] = i

    # Overlay plot: generated vs true samples
    plt.figure(figsize=plot_size)
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1],
                s=size * 0.8, alpha=0.8, c="black")
    plt.scatter(data_samples[:, 0], data_samples[:, 1],
                s=size, alpha=1, c="blue")
    plt.legend(["generated", "true"])
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Generated samples colored by condition
    plt.figure(figsize=plot_size)
    plt.title("Generated samples")
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1],
                s=size, c=color)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # True samples colored by condition
    plt.figure(figsize=plot_size)
    plt.title("Data samples")
    plt.scatter(data_samples[:, 0], data_samples[:, 1],
                s=size, c=color)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_densities(data_samples, densities, size=1, plot_size=(6, 6)):
    """
    Plot density values over 2D data samples.

    Args:
        data_samples (np.ndarray): Sample coordinates of shape (n_samples, 2).
        densities (np.ndarray): Density values corresponding to each sample.
        size (float, optional): Marker size. Defaults to 1.
        plot_size (tuple, optional): Figure size (width, height). Defaults to (6, 6).

    Returns:
        None
    """
    plt.figure(figsize=plot_size)
    plt.title("Density estimation")
    plt.scatter(data_samples[:, 0], data_samples[:, 1],
                s=size, c=densities, cmap="viridis")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_simulation_scanpy(generated_samples, data_samples, condition, cond_dims,
                           num_cond, subsample_size=None, size=1, figsize=(6, 6)):
    """
    Compare generated and real samples using UMAP visualization in Scanpy.

    The function:
    - Constructs categorical condition labels from one-hot encodings.
    - Combines real and generated samples into a single AnnData object.
    - Computes neighbors and UMAP embedding.
    - Plots comparisons by data type and condition.

    Args:
        generated_samples (np.ndarray): Generated data of shape (n_gen, n_features).
        data_samples (np.ndarray): Real data of shape (n_real, n_features).
        condition (np.ndarray): One-hot encoded condition matrix.
        cond_dims (list[int]): List of dimensions per condition group.
        num_cond (int): Number of condition groups.
        subsample_size (int, optional): Number of cells to subsample. Defaults to None.
        size (float, optional): Marker size in UMAP plots. Defaults to 1.
        figsize (tuple, optional): Figure size. Defaults to (6, 6).

    Returns:
        None
    """
    labels = []
    start = 0

    # Decode one-hot condition blocks into integer labels
    for i in range(num_cond):
        labels.append(np.argmax(condition[:, start:start + cond_dims[i]], axis=1))
        start += cond_dims[i]

    labels = np.stack(labels, axis=1)
    labels = ["_".join(map(str, list(label))) for label in labels]

    # Construct AnnData object
    obs = {
        "data_type": ["Real"] * data_samples.shape[0] +
                     ["Generated"] * generated_samples.shape[0],
        "condition": labels + labels
    }

    adata = ad.AnnData(
        X=np.concatenate([data_samples, generated_samples], axis=0),
        obs=obs
    )

    # Store PCA representation
    adata.obsm["X_pca"] = adata.X.copy()

    if subsample_size is not None:
        sc.pp.subsample(adata, subsample_size)

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    plt.rcParams['figure.figsize'] = figsize

    sc.pl.umap(adata, color="data_type", s=size, title="Comparison")
    sc.pl.umap(adata[adata.obs.data_type == "Generated"],
               color="condition", s=size, title="Generated")
    sc.pl.umap(adata[adata.obs.data_type == "Real"],
               color="condition", s=size, title="Real")


def plot_classification_scanpy(data_samples, color, color_orig, entropy,
                               subsample_size=None, size=1, figsize=(6, 6)):
    """
    Visualize classification results using Scanpy UMAP.

    Args:
        data_samples (np.ndarray): Data matrix of shape (n_samples, n_features).
        color (array-like): Predicted labels.
        color_orig (array-like): True labels.
        entropy (array-like): Entropy values per sample.
        subsample_size (int, optional): Number of cells to subsample. Defaults to None.
        size (float, optional): Marker size. Defaults to 1.
        figsize (tuple, optional): Figure size. Defaults to (6, 6).

    Returns:
        None
    """
    obs = {
        "color": list(map(str, color)),
        "color_orig": list(map(str, color_orig)),
        "entropy": entropy
    }

    adata = ad.AnnData(X=data_samples, obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()

    plt.rcParams['figure.figsize'] = figsize

    if subsample_size is not None:
        sc.pp.subsample(adata, subsample_size)

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color="color", s=size, title="Generated")
    sc.pl.umap(adata, color="color_orig", s=size, title="True")
    sc.pl.umap(adata, color="entropy", s=size,
               cmap="viridis", title="Entropy")


def plot_densities_scanpy(data_samples, densities, subsample_size=None,
                          size=1, figsize=(6, 6), title="Density", **kwargs):
    """
    Visualize densities on a UMAP embedding using Scanpy.

    Args:
        data_samples (np.ndarray): Data matrix.
        densities (array-like): Density values per sample.
        subsample_size (int, optional): Number of cells to subsample. Defaults to None.
        size (float, optional): Marker size. Defaults to 1.
        figsize (tuple, optional): Figure size. Defaults to (6, 6).
        title (str, optional): Plot title. Defaults to "Density".
        **kwargs: Additional arguments passed to sc.pl.umap().

    Returns:
        None
    """
    obs = {"density": densities}
    adata = ad.AnnData(X=data_samples, obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()

    plt.rcParams['figure.figsize'] = figsize

    if subsample_size is not None:
        sc.pp.subsample(adata, subsample_size)

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color="density", s=size,
               cmap="viridis", title=title, **kwargs)


def plot_density_ratios_scanpy(data_samples, densities, mask,
                               subsample_size=None, cmap="viridis",
                               size=1, figsize=(6, 6),
                               title="Density ratio", **kwargs):
    """
    Plot density ratios on UMAP, masking selected samples.

    Args:
        data_samples (np.ndarray): Data matrix.
        densities (array-like): Density ratio values.
        mask (np.ndarray): Boolean mask indicating valid samples.
        subsample_size (int, optional): Number of cells to subsample.
        cmap (str, optional): Colormap name. Defaults to "viridis".
        size (float, optional): Marker size.
        figsize (tuple, optional): Figure size.
        title (str, optional): Plot title.
        **kwargs: Additional arguments for sc.pl.umap().

    Returns:
        None
    """
    densities[~mask] = None

    obs = {"density": densities}
    adata = ad.AnnData(X=data_samples, obs=obs)
    adata.obsm["X_pca"] = adata.X.copy()

    plt.rcParams['figure.figsize'] = figsize

    if subsample_size is not None:
        sc.pp.subsample(adata, subsample_size)

    sc.pp.neighbors(adata)
    sc.tl.umap(adata)

    sc.pl.umap(adata, color="density", s=size,
               cmap=cmap, title=title, **kwargs)


def plot_boxplot_comparison(data1, data2, label1, label2):
    """
    Create a boxplot comparison between two datasets.

    Args:
        data1 (list or array-like): First dataset.
        data2 (list or array-like): Second dataset.
        label1 (str): Label for first dataset.
        label2 (str): Label for second dataset.

    Returns:
        None
    """
    df = pd.DataFrame({
        'value': data1 + data2,
        'group': [label1] * len(data1) + [label2] * len(data2)
    })

    sns.boxplot(x='group', y='value', data=df)
    plt.title('Boxplot comparison')
    plt.show()


def plot_density_ratios_scanpy_new(adata, densities, mask,
                                   cmap="viridis", size=1,
                                   figsize=(6, 6),
                                   title="Density ratio", **kwargs):
    """
    Plot density ratios directly on an existing AnnData object.

    Args:
        adata (AnnData): Precomputed AnnData object with UMAP embedding.
        densities (array-like): Density ratio values.
        mask (np.ndarray): Boolean mask indicating valid samples.
        cmap (str, optional): Colormap name.
        size (float, optional): Marker size.
        figsize (tuple, optional): Figure size.
        title (str, optional): Plot title.
        **kwargs: Additional arguments for sc.pl.umap().

    Returns:
        None
    """
    densities[~mask] = None
    adata.obs["density"] = densities

    plt.rcParams['figure.figsize'] = figsize

    sc.pl.umap(adata, color="density",
               s=size, cmap=cmap,
               title=title, **kwargs)
    