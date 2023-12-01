import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from bmm import BMM

matplotlib.rc("text", usetex=True)

PLOT_DIR = "./plots/"
TEST_DOC_ID = 2001


if __name__ == "__main__":
    data = sio.loadmat("kos_doc_data.mat")
    # 3 cols: doc_num, word_index, word_count
    A = np.array(data["A"])  # training word counts
    B = data["B"]  # testing word counts
    V = data["V"]

    W = np.max([np.max(A[:, 1]), np.max(B[:, 1])])  # unique words
    D = np.max(A[:, 0])  # number of documents in A

    word_counts = np.zeros(W, dtype=int)  # word counts over all documents

    print("Get word counts in docs")
    for doc_num in tqdm(range(D)):
        # all rows of A about document doc_num
        doc_indices = np.where(A[:, 0] == doc_num + 1)
        test_doc_word_indices = np.array(A[doc_indices, 1])

        word_counts[test_doc_word_indices - 1] += np.array(A[doc_indices, 2])

    total_words = np.sum(word_counts).astype(int)

    print(f"W={W} distinct words")
    print(f"N={total_words} total words")

    GIBBS_ITERATIONS = 50
    NUM_CLUSTERS = 20
    ALPHA = 10
    GAMMA = 0.1

    SEED = 10
    np.random.seed(SEED)

    perplexity, swk, sk_docs_over_time = BMM(
        A, B, NUM_CLUSTERS, ALPHA, GAMMA, GIBBS_ITERATIONS
    )

    print(f"Perplexity (Gibbs): p = {perplexity}")

    NUM_POINTS = 50
    CUTOFF = 0.05

    iterations = range(0, NUM_POINTS + 1)
    d = np.sum(sk_docs_over_time[:, 0])

    for cluster in range(NUM_CLUSTERS):
        vals = sk_docs_over_time[cluster, 0 : NUM_POINTS + 1] / d
        mean = np.mean(vals)
        label = str(cluster + 1) if mean > CUTOFF else ""
        plt.plot(iterations, vals, label=label)

    plt.grid(alpha=0.4)
    plt.title(rf"Categorisation Progress $\vert$ Seed = {SEED}")
    plt.xlabel("Iteration")
    plt.ylabel("Proportion in each category")
    plt.xlim(left=0)
    # plt.legend(loc="upper left")

    plt.savefig(
        PLOT_DIR + f"d_progress_seed_{SEED}.png",
        dpi=500,
        format="png",
        bbox_inches="tight",
    )
    # plt.show()
