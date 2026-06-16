from slimder_man.compression.planner import progressive_plan


def test_progressive_plans():
    depth = progressive_plan("depth_first", 2, 400_000_000_000, [0.1, 0.9], 48, 12, 2048, 1536)
    assert depth[0].remove_last_n_layers == 6 and depth[0].hidden_size == 2048
    assert depth[1].remove_last_n_layers == 12 and depth[1].hidden_size == 1536
    width = progressive_plan("width_first", 2, 400, [0.1, 0.9], 48, 12, 2048, 1536)
    assert width[0].hidden_size == 1792 and width[0].remove_last_n_layers == 0
    joint = progressive_plan("joint", 2, 400, [0.1, 0.9], 48, 12, 2048, 1536)
    assert joint[0].remove_last_n_layers == 6 and joint[0].hidden_size == 1792
    assert depth[0].tokens == 40_000_000_000 and depth[1].tokens == 360_000_000_000
    assert depth[0].routed_experts is None
