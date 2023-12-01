import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("text", usetex=True)

PLOT_DIR = "./plots/"

if __name__ == "__main__":
    data = sio.loadmat("kos_doc_data.mat")
    # 3 cols: doc_num, word_index, word_count
    A = np.array(data["A"])  # training word counts
    B = data["B"]  # testing word counts
    V = data["V"]

    W = np.max([np.max(A[:, 1]), np.max(B[:, 1])])  # unique words
    D = np.max(A[:, 0])  # number of documents in A

    word_counts = np.zeros(W, dtype=int)  # word counts over all documents

    for doc_num in range(D):
        # all rows of A about document doc_num
        doc_indices = np.where(A[:, 0] == doc_num + 1)
        word_indices = np.array(A[doc_indices, 1])

        word_counts[word_indices - 1] += np.array(A[doc_indices, 2])

    total_words = np.sum(word_counts).astype(int)

    print(f"W={W} distinct words")
    print(f"N={total_words} total words")

    top, top_axs = plt.subplots(1, 4, constrained_layout=True, figsize=(14, 3.2))
    bot, bot_axs = plt.subplots(1, 1, constrained_layout=True, figsize=(14, 3.2))

    for ax, alpha in enumerate([0.1, 1, 10, 100]):
        plt.figure(top)

        prior_counts = alpha * np.ones(W)
        posterior_counts = word_counts + prior_counts
        total_count = np.sum(posterior_counts)

        posterior_frequencies = np.true_divide(posterior_counts, total_count)

        NUM_TO_SHOW = 20
        top_words_indices = np.argsort(posterior_frequencies)[::-1][:NUM_TO_SHOW]
        top_frequencies = posterior_frequencies[top_words_indices]
        top_words = [word[0] for word in V[top_words_indices, 0]]

        top_axs[ax].grid(alpha=0.4, zorder=0)

        top_axs[ax].barh(top_words[::-1], top_frequencies[::-1], zorder=3)
        top_axs[ax].set_title(rf"\textbf{{Alpha = {alpha}}}")
        top_axs[ax].set_ylabel(r"\textbf{Word}")
        top_axs[ax].set_xlabel(r"\textbf{Probability}")

        plt.figure(bot)

        bottom_words_indices = np.argsort(posterior_frequencies)[:NUM_TO_SHOW]
        bottom_frequencies = posterior_frequencies[bottom_words_indices]
        bottom_words = [word[0] for word in V[bottom_words_indices, 0]]

        bot_axs[ax].grid(alpha=0.4, zorder=0)

        bot_axs[ax].barh(bottom_words, bottom_frequencies, zorder=3)
        bot_axs[ax].set_title(rf"\textbf{{Alpha = {alpha}}}")
        bot_axs[ax].set_ylabel(r"\textbf{Word}")
        bot_axs[ax].set_xlabel(r"\textbf{Probability}")

    plt.figure(top)
    plt.suptitle(r"\textbf{Top 20 $\vert$ Posterior Pred. Dist.}")
    plt.savefig(
        PLOT_DIR + f"b_pred_histogram_{alpha}.png",
        dpi=500,
        format="png",
        bbox_inches="tight",
    )

    plt.figure(bot)
    plt.suptitle(r"\textbf{Bottom 20 $\vert$ Posterior Pred. Dist.}")
    plt.savefig(
        PLOT_DIR + f"b_pred_histogram_bottom_{alpha}.png",
        dpi=500,
        format="png",
        bbox_inches="tight",
    )
    plt.show()
