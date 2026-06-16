from __future__ import annotations

import torch


def router_rows_for_merge(original_rows: torch.Tensor, s_keep: list[int], s_base: list[int], strategy: str = "base") -> torch.Tensor:
    if strategy != "base":
        raise ValueError("paper_faithful mode requires router_row_strategy=base")
    return original_rows.index_select(0, torch.tensor(s_keep + s_base, dtype=torch.long, device=original_rows.device)).detach().clone()
