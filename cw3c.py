import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm

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

    for doc_num in range(D):
        # all rows of A about document doc_num
        doc_indices = np.where(A[:, 0] == doc_num + 1)
        test_doc_word_indices = np.array(A[doc_indices, 1])

        word_counts[test_doc_word_indices - 1] += np.array(A[doc_indices, 2])

    total_words = np.sum(word_counts).astype(int)

    print(f"W={W} distinct words")
    print(f"N={total_words} total words")

    ALPHA = 0.1

    prior_counts = ALPHA * np.ones(W)

    posterior_counts = word_counts + prior_counts
    total_count = np.sum(posterior_counts)

    posterior_frequencies = np.true_divide(posterior_counts, total_count)

    NUM_TO_SHOW = 20
    t_s_top_words_indices = np.argsort(posterior_frequencies)[::-1][:NUM_TO_SHOW]
    t_s_top_frequencies = posterior_frequencies[t_s_top_words_indices]
    t_s_top_words = [word[0] for word in V[t_s_top_words_indices, 0]]

    test_doc_indices = np.where(B[:, 0] == TEST_DOC_ID)

    test_doc_word_counts = np.zeros(W, dtype=int)
    test_doc_word_indices = np.array(B[test_doc_indices, 1])

    test_doc_word_counts[test_doc_word_indices - 1] += np.array(B[test_doc_indices, 2])

    test_doc_total_words = np.sum(test_doc_word_counts)

    log_posterior_frequencies = np.log(posterior_frequencies)
    test_doc_log_probability = np.dot(test_doc_word_counts, log_posterior_frequencies)
    test_doc_perplexity = np.exp(-1 * test_doc_log_probability / test_doc_total_words)

    print(f"Test Doc (ID: {TEST_DOC_ID})")
    print(f"\tLog Probability l = {test_doc_log_probability}")
    print(f"\tTotal Words N = {test_doc_total_words}")
    print(f"\tPerplexity = {test_doc_perplexity}")

    num_docs_test_set = np.max(B[:, 0])  # highest document index

    test_set_word_counts = np.zeros(W, dtype=int)

    for doc in tqdm(range(num_docs_test_set)):
        test_set_doc_indices = np.where(B[:, 0] == doc + 1)
        # fix stupid matlab indexing
        test_set_doc_word_indices = np.array(B[test_set_doc_indices, 1])
        temp = np.array(B[test_set_doc_indices, 2])
        test_set_word_counts[test_set_doc_word_indices - 1] += temp

    test_set_total_words = np.sum(test_set_word_counts)
    test_set_word_frequencies = np.true_divide(
        test_set_word_counts, test_doc_total_words
    )

    test_set_log_probability = np.dot(test_set_word_counts, log_posterior_frequencies)

    test_set_perplexity = np.exp(-1 * test_set_log_probability / test_set_total_words)

    print("Test Set")
    print(f"\tNum. Distinct Words: W = {W}")
    print(f"\tNum. Words Total: N = {test_set_total_words}")
    print(f"\tPerplexity: p = {test_set_perplexity}")

    NUM_TO_SHOW = 20
    t_s_top_words_indices = np.argsort(posterior_frequencies)[::-1][:NUM_TO_SHOW]
    t_s_top_frequencies = posterior_frequencies[t_s_top_words_indices]
    t_s_top_words = [word[0] for word in V[t_s_top_words_indices, 0]]

    plt.grid(alpha=0.4, zorder=0)
    plt.barh(t_s_top_words[::-1], t_s_top_frequencies[::-1], zorder=3)
    plt.title(rf"\textbf{{Top words in Test Set $\vert~\alpha$ = {ALPHA}}}")
    plt.ylabel(r"\textbf{Word}")
    plt.xlabel(r"\textbf{Frequency}")

    plt.savefig(
        PLOT_DIR + "c_top_words.png",
        dpi=500,
        format="png",
        bbox_inches="tight",
    )
    plt.show()
