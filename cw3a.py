import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("text", usetex=True)

plot_dir = "./plots/"

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

    print(word_counts[:5])

    total_words = np.sum(word_counts).astype(int)
    word_frequencies = np.true_divide(word_counts, total_words)

    print(f"W={W} distinct words")
    print(f"N={total_words} total words")

    num_to_show = 20
    top_words_indices = np.argsort(word_frequencies)[::-1][:num_to_show]
    top_frequencies = word_frequencies[top_words_indices]
    top_words = [word[0] for word in V[top_words_indices, 0]]

    plt.grid(alpha=0.4, zorder=0)

    plt.barh(top_words[::-1], top_frequencies[::-1], zorder=3)
    plt.title(r"\textbf{20 Most Prevalent Words in Training Set}")
    plt.ylabel(r"\textbf{Word}")
    plt.xlabel(r"\textbf{Frequency of Occurence}")
    plt.savefig(
        plot_dir + "a_histogram.png", dpi=500, format="png", bbox_inches="tight"
    )

    plt.figure()

    bottom_words_indices = np.argsort(word_frequencies)[:num_to_show]
    bottom_frequencies = word_frequencies[bottom_words_indices]
    bottom_words = [word[0] for word in V[bottom_words_indices, 0]]

    plt.grid(alpha=0.4, zorder=0)

    plt.barh(bottom_words, bottom_frequencies, zorder=3)
    plt.title(r"\textbf{20 Least Prevalent Words in Training Set}")
    plt.ylabel(r"\textbf{Word}")
    plt.xlabel(r"\textbf{Frequency of Occurence}")
    plt.savefig(
        plot_dir + "a_histogram_bottom.png", dpi=500, format="png", bbox_inches="tight"
    )
    plt.show()
