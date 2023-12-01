import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from tqdm import tqdm

from lda import LDA

matplotlib.rc("text", usetex=True)

PLOT_DIR = "./plots/"


def get_thetas_at_iter(arg_list_skd, alpha, docid, gibbs_iter):
    skd_iter = arg_list_skd[gibbs_iter]
    skd_doc_iter = skd_iter[:, docid]
    skd_doc_iter_alpha = skd_doc_iter + alpha
    thetas_iter = skd_doc_iter_alpha / np.sum(skd_doc_iter_alpha)
    return thetas_iter


def get_thetas_per_iter(arg_list_skd, alpha, docid, total_iters):
    thetas_per_iter = []
    for iteration in range(total_iters):
        thetas_iter = get_thetas_at_iter(arg_list_skd, alpha, docid, iteration)
        thetas_per_iter.append(thetas_iter)

    return np.array(thetas_per_iter)


def plot_thetas(arg_list_skd, alpha, docid, total_iters):
    thetas_per_iter = get_thetas_per_iter(arg_list_skd, alpha, docid, total_iters)
    thetas_per_iter = np.transpose(thetas_per_iter)

    plt.figure(figsize=(6.4, 4.5))

    x_s = range(total_iters)
    for k in range(thetas_per_iter.shape[0]):
        plt.plot(x_s, thetas_per_iter[k], label=rf"$\theta_{{{k}}}$")

    plt.xlabel("Gibbs iteration", fontsize=13)
    plt.ylabel("Posterior probability", fontsize=13)
    plt.xlim(left=0)
    plt.title(f"Topic Allocation for Document {docid}", fontsize=15)
    plt.grid(alpha=0.4)

    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))


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

    np.random.seed(0)

    NUM_CLUSTERS = 20  # K
    ALPHA = 0.1
    GAMMA = 0.1
    NUM_ITERS = 50

    # print("RUNNING LDA COMMAND - WILL TAKE F**KING AGES")
    # lda_perplexity_, lda_swk_, list_skd_, entropies_ = LDA(
    #     A, B, NUM_CLUSTERS, ALPHA, GAMMA, NUM_ITERS
    # )

    # np.save("data/lda_perp.npy", lda_perplexity_)
    # np.save("data/lda_swk.npy", lda_swk_)
    # np.save("data/list_skd.npy", list_skd_)
    # np.save("data/entropies.npy", entropies_)

    lda_perplexity = np.load("data/lda_perp.npy")
    lda_swk = np.load("data/lda_swk.npy")
    list_skd = np.load("data/list_skd.npy")
    entropies = np.load("data/entropies.npy")

    print("Test Set")
    print(f"\tPerplexity: p = {lda_perplexity}")
    print("\tFinal Entropies: E =")
    print("\t" + str(entropies[-1]).replace("\n", "\n\t"))

    for doc_id in [11, 15, 1000]:
        plot_thetas(list_skd, alpha=0.1, docid=doc_id, total_iters=NUM_ITERS)

        plt.savefig(
            PLOT_DIR + f"e_thetas_doc_{doc_id}.png",
            dpi=500,
            format="png",
            bbox_inches="tight",
        )
    plot_entropies = entropies.T

    plt.figure(figsize=(7.4, 5.5))

    for i in range(plot_entropies.shape[0]):
        x = range(plot_entropies.shape[1])
        plt.plot(x, plot_entropies[i], label=f"Topic {i}")

    plt.xlabel("Gibbs iteration", fontsize=13)
    plt.ylabel("Word entropy (nats)", fontsize=13)
    plt.title("Word Entropies", fontsize=15)
    plt.xlim(left=0)
    plt.grid(alpha=0.4)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.savefig(
        PLOT_DIR + "e_entropies.png",
        dpi=500,
        format="png",
        bbox_inches="tight",
    )

    # plt.show()
