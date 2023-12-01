import numpy as np
import scipy.io as sio
from tqdm import tqdm

from sampleDiscrete import sampleDiscrete


def BMM(
    training_data, test_data, num_mixture_components, alpha, gamma, num_iters_gibbs=10
):
    """

    :param A: Training data [D, 3]
    :param B: Test Data [D, 3]
    :param K: number of mixture components
    :param alpha: parameter of the Dirichlet over mixture components
    :param gamma: parameter of the Dirichlet over words
    :return: test perplexity and multinomial weights over words
    """
    total_words = np.max(
        [np.max(training_data[:, 1]), np.max(test_data[:, 1])]
    )  # total number of unique words
    total_docs = np.max(training_data[:, 0])  # number of documents in A

    # Initialization: assign each document a mixture component at random
    sd = np.floor(num_mixture_components * np.random.rand(total_docs)).astype(
        int
    )  # mixture component assignment
    bmm_swk = np.zeros(
        (total_words, num_mixture_components)
    )  # K multinomials over W unique words
    sk_docs = np.zeros(
        (num_mixture_components, 1), dtype=int
    )  # number of documents assigned to each mixture
    print("Populate the count matrices by looping over documents")
    for d in tqdm(range(total_docs)):
        training_documents = np.where(
            training_data[:, 0] == d + 1
        )  # get all occurrences of document d in the training data
        w = np.array(
            training_data[training_documents, 1]
        )  # number of unique words in document d
        c = np.array(
            training_data[training_documents, 2]
        )  # counts of words in document d
        k = sd[d]  # document d is in mixture k
        bmm_swk[w - 1, k] += c  # number of times w is assigned to component k
        sk_docs[k] += 1

    sk_words = np.sum(
        bmm_swk, axis=0
    )  # number of words assigned to mixture k over all docs

    sk_docs_history = np.zeros((num_mixture_components, num_iters_gibbs + 1), dtype=int)
    sk_docs_history[:, 0] = np.copy(sk_docs)[:, 0]

    print("Perform Gibbs sampling through all documents and words")
    for iteration in tqdm(range(num_iters_gibbs)):
        for d in range(total_docs):
            training_documents = np.where(
                training_data[:, 0] == d + 1
            )  # get all occurrences of document d in trh training data
            w = training_data[
                training_documents, 1
            ]  # number of unique words in document d
            c = training_data[training_documents, 2]  # counts of words in document d
            old_class = sd[d]  # document d is in mixture k
            # remove document from counts
            bmm_swk[
                w - 1, old_class
            ] -= c  # decrease number of times w is assigned to component k
            sk_docs[old_class] -= 1  # remove document d from count of docs
            sk_words[old_class] -= np.sum(c)  # remove word counts from mixture
            # resample class of document
            lb = np.zeros(
                num_mixture_components
            )  # log probability of doc d under mixture component k

            for k in range(num_mixture_components):
                ll = np.dot(
                    np.log(bmm_swk[w - 1, k] + gamma)
                    - np.log(sk_words[k] + gamma * total_words),
                    c.T,
                )

                lb[k] = np.log(sk_docs[k] + alpha) + ll
            b = np.exp(
                lb - np.max(lb)
            )  # exponentiation of log probability plus constant
            kk = sampleDiscrete(
                b, np.random.rand()
            )  # sample from (un-normalized) multinomial distribution
            # update counts based on new class assignment
            bmm_swk[w - 1, kk] += c  # number of times w is assigned to component k
            sk_docs[kk] += 1
            sk_words[kk] += np.sum(c)
            sd[d] = kk

        # add on to history
        sk_docs_history[:, iteration + 1] = np.copy(sk_docs)[:, 0]

    # test documents
    lp = 0
    nd = 0
    unique_docs_in_b = np.unique(test_data[:, 0])
    print("Loop over test docs")
    for doc in tqdm(unique_docs_in_b):
        test_docs = np.where(test_data[:, 0] == doc)
        w = test_data[test_docs, 1]  # unique words in doc d
        c = test_data[test_docs, 2]  # counts
        z = np.log(sk_docs + alpha) - np.log(np.sum(sk_docs + alpha))
        for k in range(num_mixture_components):
            b = (bmm_swk[:, k] + gamma) / (sk_words[k] + gamma * total_words)
            z[k] += np.dot(c, np.log(b[w - 1]).T)[0]  # probability for doc d
        lp += np.log(np.sum(np.exp(z - np.max(z)))) + np.max(
            z
        )  # log-sum-exp to compute normalization constant
        nd += np.sum(c)

    bmm_perplexity = np.exp(-lp / nd)  # perplexity

    return bmm_perplexity, bmm_swk, sk_docs_history


if __name__ == "__main__":
    np.random.seed(1)
    # load data
    data = sio.loadmat("kos_doc_data.mat")
    A = np.array(data["A"])
    B = data["B"]
    V = data["V"]
    K = 20  # number of clusters
    ALPHA = 10  # parameter of the Dirichlet over mixture components
    GAMMA = 0.1  # parameter of the Dirichlet over words
    perplexity, swk, _ = BMM(A, B, K, ALPHA, GAMMA)
    print(perplexity)

    NUM_TO_DISPLAY = 20
    indices = np.argsort(-swk, axis=0)
    indices = indices[:NUM_TO_DISPLAY, :]
    top_words = V[indices]
    for topic in top_words[:, :, 0].T:
        print("\n")
        for word in topic:
            print(word[0])
