import scipy.io as sio
import numpy as np
from scipy.sparse import coo_matrix as sparse
from tqdm import tqdm

from sampleDiscrete import sampleDiscrete


def LDA(training_data, test_data, num_mixture_comp, alpha, gamma, num_gibbs_iters=10):
    """
    Latent Dirichlet Allocation

    :param A: Training data [D, 3]
    :param B: Test Data [D, 3]
    :param K: number of mixture components
    :param alpha: parameter of the Dirichlet over mixture components
    :param gamma: parameter of the Dirichlet over words
    :return: perplexity, multinomial over words
    """
    W = np.max(
        [np.max(training_data[:, 1]), np.max(test_data[:, 1])]
    )  # total number of unique words
    D = np.max(training_data[:, 0])  # number of documents in A

    # A's columns are doc_id, word_id, count
    swd = sparse(
        (training_data[:, 2], (training_data[:, 1] - 1, training_data[:, 0] - 1))
    ).tocsr()
    Swd = sparse((test_data[:, 2], (test_data[:, 1] - 1, test_data[:, 0] - 1))).tocsr()

    # Initialization
    skd = np.zeros(
        (num_mixture_comp, D)
    )  # count of word assignments to topics for document d
    _swk = np.zeros(
        (W, num_mixture_comp)
    )  # unique word topic assignment counts across all documents

    s = []  # each element of the list corresponds to a document
    r = 0
    for d in tqdm(range(D), desc="Documents"):  # iterate over the documents
        z = np.zeros(
            (W, num_mixture_comp)
        )  # unique word topic assignment counts for doc d
        words_in_doc_d = training_data[np.where(training_data[:, 0] == d + 1), 1][0] - 1
        for w in words_in_doc_d:  # loop over the unique words in doc d
            c = swd[w, d]  # number of occurrences for doc d
            for _ in range(c):  # assign each occurrence of word w to a doc at random
                k = np.floor(num_mixture_comp * np.random.rand())
                z[w, int(k)] += 1
                r += 1
        skd[:, d] = np.sum(z, axis=0)  # number of words in doc d assigned to each topic
        _swk += z  # unique word topic assignment counts across all documents
        s.append(sparse(z))  # sparse representation: z contains many zero entries

    sk = np.sum(skd, axis=1)  # word to topic assignment counts accross all documents

    skd_list = []
    entropies = []
    # This makes a number of Gibbs sampling sweeps through all docs and words, it may take a bit to run
    print("Gibbs sample over all docs and words")
    for _ in tqdm(range(num_gibbs_iters), desc="Each Iter."):
        skd_list.append(np.copy(skd))
        for d in tqdm(range(D), desc="Over Docs", leave=False):
            z = s[d].todense()  # unique word topic assigmnet counts for document d
            words_in_doc_d = (
                training_data[np.where(training_data[:, 0] == d + 1), 1][0] - 1
            )
            for w in words_in_doc_d:  # loop over unique words in doc d
                # number of times word w is assigned to each topic in doc d
                a = z[w, :].copy()
                # topics with non-zero word counts for word w in doc d
                non_zero_indices = np.where(a > 0)[1]
                np.random.shuffle(non_zero_indices)
                for k in non_zero_indices:
                    k = int(k)
                    for i in range(int(a[0, k])):  # loop over counts for topic k
                        z[w, k] -= 1  # remove word from count matrices
                        _swk[w, k] -= 1
                        sk[k] -= 1
                        skd[k, d] -= 1
                        b = (
                            (alpha + skd[:, d])
                            * (gamma + _swk[w, :])
                            / (W * gamma + sk)
                        )
                        kk = sampleDiscrete(
                            b, np.random.rand()
                        )  # Gibbs sample new topic assignment
                        z[w, kk] += 1  # add word with new topic to count matrices
                        _swk[w, kk] += 1
                        sk[kk] += 1
                        skd[kk, d] += 1

            s[d] = sparse(z)  # store back into sparse structure

        betas = np.array(
            [
                (_swk[:, topic_ID] + gamma) / (np.sum(_swk[:, topic_ID] + gamma))
                for topic_ID in range(num_mixture_comp)
            ]
        )

        # Added code to calculate word entropies per iteration
        aux_entropies = []
        for k in range(num_mixture_comp):
            word_entropy = 0

            for w in range(W):
                word_entropy += -1 * betas[k, w] * np.log2(betas[k, w])
            aux_entropies.append(word_entropy)
        entropies.append(aux_entropies)

    # compute the perplexity for all words in the test set B
    # We need the new Skd matrix, derived from corpus B
    lp, nd = 0, 0
    unique_docs_in_b = np.unique(test_data[:, 0])
    print("Loop over Test Set")
    for d in tqdm(
        unique_docs_in_b, desc="Loop test set"
    ):  # loop over all documents in B
        # randomly assign topics to each word in test document d
        z = np.zeros((W, num_mixture_comp))
        words_in_d = test_data[np.where(test_data[:, 0] == d), 1][0] - 1
        for w in words_in_d:  # w are the words in doc d
            c = Swd[w, d - 1]
            for i in range(c):
                k = np.floor(num_mixture_comp * np.random.rand())
                z[w, int(k)] += 1

        Skd = np.sum(z, axis=0)
        # perform some iterations of Gibbs sampling for test document d

        for _ in range(num_gibbs_iters):
            for w in words_in_d:  # w are the words in doc d
                a = z[
                    w, :
                ].copy()  # number of times word w is assigned to each topic in doc d
                non_zero_indices = np.where(a > 0)[
                    0
                ]  # topics with non-zero word counts for word w in doc d
                np.random.shuffle(non_zero_indices)
                for k in non_zero_indices:
                    k = int(k)
                    for i in range(int(a[k])):
                        z[w, k] -= 1  # remove word from count matrix for doc d
                        Skd[k] -= 1
                        b = (alpha + Skd) * (gamma + _swk[w, :]) / (W * gamma + sk)
                        kk = sampleDiscrete(b, np.random.rand())
                        z[
                            w, kk
                        ] += 1  # add word with new topic to count matrix for doc d
                        Skd[kk] += 1
        b1 = ((alpha + Skd) / np.sum(alpha + Skd))[:, None]
        b2 = (gamma + _swk) / (W * gamma + sk)
        b = np.matmul(b2, b1)
        words_and_counts = test_data[np.where(test_data[:, 0] == d), 1:][0]
        lp += np.dot(
            np.log(b[words_and_counts[:, 0] - 1]).T, words_and_counts[:, 1]
        )  # log probability, doc d
        nd += np.sum(words_and_counts[:, 1])  # number of words, doc d

    _perplexity = np.exp(-lp / nd)  # perplexity

    return _perplexity, _swk, np.array(skd_list), np.array(entropies)


if __name__ == "__main__":
    np.random.seed(0)
    # load data
    data = sio.loadmat("kos_doc_data.mat")
    A = np.array(data["A"])
    B = data["B"]
    V = data["V"]

    K = 20  # number of clusters
    ALPHA = 0.1  # parameter of the Dirichlet over mixture components
    GAMMA = 0.1  # parameter of the Dirichlet over words

    perplexity, swk, _, _ = LDA(A, B, K, ALPHA, GAMMA)
    print(perplexity)
    NUM_TO_DISPLAY = 20
    indices = np.argsort(-swk, axis=0)
    indices = indices[:NUM_TO_DISPLAY, :]
    top_words = V[indices]
    for topic in top_words[:, :, 0].T:
        print("\n")
        for word in topic:
            print(word[0])
