import torch

from slimder_man.calibration.stats import cosine_similarity_from_matrix


def test_expert_similarity_properties():
    matrix = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    sim = cosine_similarity_from_matrix(matrix)
    assert sim[0, 2] == 1
    assert sim[0, 1] == 0
    assert torch.allclose(sim, sim.T)
    assert sim[0, 0] == 1
