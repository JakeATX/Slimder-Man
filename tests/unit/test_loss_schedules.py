from slimder_man.distill.schedules import cosine_schedule, global_cosine_lr, linear_schedule


def test_lambda_and_beta_schedules():
    assert linear_schedule(1.0, 0.75, 0, 3) == 1.0
    assert linear_schedule(1.0, 0.75, 1, 3) == 0.875
    assert linear_schedule(1.0, 0.75, 2, 3) == 0.75
    vals = [cosine_schedule(0.3, 0.1, i, 5) for i in range(5)]
    assert vals[0] == 0.3
    assert vals[-1] == 0.1
    assert vals == sorted(vals, reverse=True)


def test_global_lr_does_not_restart():
    lr1 = global_cosine_lr(1.0, 0.1, 2, 3, 10)
    lr2 = global_cosine_lr(1.0, 0.1, 2, 4, 10)
    assert lr2 <= lr1
