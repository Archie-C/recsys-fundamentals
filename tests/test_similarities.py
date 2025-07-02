import numpy as np

from src.utils.similarities import adjusted_cosine_similarity, euclidean_similarity, hamming_similarity, jaccard_similarity, cosine_similarity, log_likelihood_similarity, manhattan_similarity, pearson_similarity, tanimoto_similarity

def test_cosine_similarity():
    a = np.array([1, 0, 1, np.nan])
    b = np.array([1, 1, 1, np.nan])
    expected = 2 / (np.sqrt(2) * np.sqrt(3))
    assert np.isclose(cosine_similarity(a, b), expected)

    a = np.array([0, 0, 0])
    b = np.array([0, 0, 0])
    assert cosine_similarity(a, b) == 0

    a = np.array([np.nan, np.nan])
    b = np.array([np.nan, np.nan])
    assert cosine_similarity(a, b) == 0

def test_euclidean_similarity():
    a = np.array([1, 2, np.nan])
    b = np.array([1, 4, np.nan])
    expected = 1 / (1 + 2)
    assert np.isclose(euclidean_similarity(a, b), expected)

    a = np.array([np.nan, np.nan])
    b = np.array([np.nan, np.nan])
    assert euclidean_similarity(a, b) == 0

    a = np.array([0, 0, 0])
    b = np.array([0, 0, 0])
    assert euclidean_similarity(a, b) == 1.0

def test_manhattan_similarity():
    a = np.array([1, 2, np.nan])
    b = np.array([1, 4, np.nan])
    expected = 1 / (1 + 2)
    assert np.isclose(manhattan_similarity(a, b), expected)

    a = np.array([np.nan, np.nan])
    b = np.array([np.nan, np.nan])
    assert manhattan_similarity(a, b) == 0

    a = np.array([0, 0, 0])
    b = np.array([0, 0, 0])
    assert manhattan_similarity(a, b) == 1.0

def test_pearson_similarity():
    a = np.array([1, 2, 3, np.nan])
    b = np.array([1, 5, 7, np.nan])
    assert np.isclose(pearson_similarity(a, b), 1.0, atol=1e-1)

    a = np.array([1, 1, 1])
    b = np.array([2, 2, 2])
    assert pearson_similarity(a, b) == 0

    a = np.array([1, np.nan, 2])
    b = np.array([2, 4, 3])
    assert -1 <= pearson_similarity(a, b) <= 1

def test_adjusted_cosine_similarity():
    users_items = np.array([
        [5, 3, np.nan],
        [4, np.nan, np.nan],
        [1, 1, 5],
        [np.nan, 1, 4]
    ])

    sim = adjusted_cosine_similarity(0, 1, users_items)
    assert -1 <= sim <= 1

    users_items = np.array([
        [5, np.nan],
        [np.nan, 4]
    ])
    assert adjusted_cosine_similarity(0, 1, users_items) == 0

def test_jaccard_similarity():
    a = np.array([1, 0, 1, 1])
    b = np.array([1, 1, 0, 1])
    assert np.isclose(jaccard_similarity(a, b), 2/4)

    a = np.array([1, 1, 1])
    b = np.array([1, 1, 1])
    assert jaccard_similarity(a, b) == 1.0

    a = np.array([0, 0, 0])
    b = np.array([0, 0, 0])
    assert jaccard_similarity(a, b) == 0.0 

    a = np.array([1, np.nan, 1])
    b = np.array([1, 0, 0])
    assert np.isclose(jaccard_similarity(a, b), 0.5)

def test_log_likelihood_similarity():

    a = np.array([1, 0, 1, 1])
    b = np.array([1, 0, 1, 1])
    assert log_likelihood_similarity(a, b) > 0

    a = np.array([1, 1, 1, 1])
    b = np.array([0, 0, 0, 0])
    assert log_likelihood_similarity(a, b) <= 0

    a = np.array([0, 0, 0, 0])
    b = np.array([1, 0, 1, 0])
    log_likelihood_similarity(a, b) 

    a = np.array([1, np.nan, 0, 1])
    b = np.array([1, 0, 0, 1])
    log_likelihood_similarity(a, b)

def test_tanimoto_similarity():
    a = np.array([1, 0, 1])
    b = np.array([1, 1, 1])
    assert np.isclose(tanimoto_similarity(a, b), 2/3)

    a = np.array([0, 0, 0])
    b = np.array([0, 0, 0])
    assert tanimoto_similarity(a, b) == 0.0

    a = np.array([1, np.nan, 1])
    b = np.array([1, 1, 0])
    assert np.isclose(tanimoto_similarity(a, b), 1/2)

def test_hamming_similarity():
    a = np.array([1, 0, 1, 1])
    b = np.array([1, 1, 0, 1])
    assert np.isclose(hamming_similarity(a, b), 0.5)

    a = np.array([1, 1, 1])
    b = np.array([1, 1, 1])
    assert hamming_similarity(a, b) == 1.0

    a = np.array([0, 0, 0])
    b = np.array([1, 1, 1])
    assert hamming_similarity(a, b) == 0.0

    a = np.array([1, np.nan, 0])
    b = np.array([1, 1, 0])
    assert np.isclose(hamming_similarity(a, b), 1.0)