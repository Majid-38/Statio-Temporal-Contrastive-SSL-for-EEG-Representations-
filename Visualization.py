import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def save_confusion(cm, labels, out_path: str, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
