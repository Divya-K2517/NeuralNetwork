import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
 
 
def plot_confusion_matrix(y_true, y_pred, class_names, title,test_acc,
                           test_loss=None, save_path=None):
    n = len(class_names)
    conf_matrix = np.zeros((n,n), dtype=int)

    for t, p in zip(y_true, y_pred):
        #if y_true is 2 and y_pred is 1, then we know that one of class 2 was predicted as 1
        #and then we increase [t][p] by 1
        conf_matrix[t][p] += 1
    
    #now make the cells show percentages
    row_sums = conf_matrix.sum(axis=1, keepdims=True) #sum of examples of a certain class
    row_sums[row_sums == 0] = 1
    conf_norm = conf_matrix / row_sums

    fig, ax = plt.subplots(figsize=(8.5, 7.5))

    #pink colormap!
    cmap = LinearSegmentedColormap.from_list("clean_blue", ["#f7fbff", "#65056F"])
    im = ax.imshow(conf_norm, cmap=cmap, vmin=0, vmax=1)

    #putting gridlines between cells
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
 
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=11, labelpad=10)
    ax.set_ylabel("True label", fontsize=11, labelpad=10)

    #annotating each cell with count and %
    for i in range(n):
        for j in range(n):
            pct = conf_norm[i, j] * 100
            count = conf_matrix[i, j]
            text_color = "white" if conf_norm[i, j] > 0.5 else "#1a1a1a"
            label = f"{count}\n{pct:.0f}%" if count > 0 else ""
            ax.text(j, i, label, ha="center", va="center",
                     fontsize=9, color=text_color, linespacing=1.3)
            
    #formatting colorbR
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fraction of true class", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    #title
    headline = f"{title}  —  Test Accuracy: {test_acc*100:.1f}%"
    if test_loss is not None:
        headline += f"   |   Loss: {test_loss:.3f}"
    ax.set_title(headline, fontsize=14, fontweight="bold", pad=16)
 
    fig.text(0.5, 0.01,
              "Neural network built from scratch in NumPy (custom forward/backprop, dropout, Adam optimizer)",
              ha="center", fontsize=8.5, color="#555555", style="italic")
 
    plt.tight_layout(rect=[0, 0.02, 1, 1])
 
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved confusion matrix figure to {save_path}")
    plt.show()
    plt.close(fig)
 
    return conf_matrix