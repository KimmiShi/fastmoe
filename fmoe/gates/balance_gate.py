r"""
Naive gate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_gate import BaseGate


class LinearTopKGate(torch.nn.Module):
    def __init__(self, model_dim, num_global_experts, k=1, fp32_gate=False, **options):
        super().__init__()
        try:
            self.wg = torch.nn.Linear(
                model_dim,
                num_global_experts,
                bias=False,
                dtype=torch.float32 if fp32_gate else None,
            )
        except:  # noqa: E722
            self.wg = torch.nn.Linear(model_dim, num_global_experts, bias=False)
        self.top_k = min(num_global_experts, int(k))
        self.fp32_gate = fp32_gate

        for opt in options:
            if opt not in ("capacity_factor", "gate_noise"):
                raise Exception(
                    "Unrecognized argument provided to Gating module: %s" % opt
                )
        self.gate_noise = options.get('gate_noise', 0)
        self.capacity_factor = options.get('capacity_factor', 0)

    def forward(self, x):
        if self.fp32_gate:
            x = x.float()
            wg = self.wg.float()
        else:
            wg = self.wg
        return wg(x)

def enforce_router_expert(scores):
    import pdb; pdb.set_trace()
    if torch.isnan(scores).int().sum() > 0:
        scores_ = scores.new_ones(scores.shape[0], scores.shape[1])
    else:
        scores_ = scores.clone()
    cap = scores_.shape[0] // scores_.shape[1]
    shuffle_idx = torch.randperm(scores_.shape[1])
    for i in range(0, scores_.shape[1]):
        aim_expert = shuffle_idx[i]
        _, aim_idx = scores_[:,aim_expert].topk(k=cap)
        scores_[aim_idx, 0 : aim_expert] = scores_[aim_idx, 0 : aim_expert] * 0
        scores_[aim_idx, aim_expert + 1 : scores_.shape[1]] = scores_[aim_idx, aim_expert + 1 : scores_.shape[1]] * 0
    topk_indices = scores_.topk(k=1,dim=1).indices
    indices_s = [x.view(-1) for x in topk_indices.chunk(1, dim=1)]
    return indices_s, indices_s[0].reshape(-1,1)


def extract_critical_fmoe(scores, top_k, loss_fn=losses.gshard_loss, capacity_factor=1.0,
                          batch_prioritized_routing=False, normalize_gate=True, group=None, enforce_router=True,
                          training=True):

    num_global_experts = int(scores.size(1))
    top_k = min(top_k, num_global_experts)

    if not enforce_router or not training:
        topk_indices = torch.topk(scores, top_k, dim=1).indices
        indices_s = [x.view(-1) for x in topk_indices.chunk(top_k, dim=1)]
    else:
        indices_s, topk_indices = enforce_router_expert(scores)
        if False:
            print(sim_test(scores))

    import pdb; pdb.set_trace()
    masks_se = [losses._one_hot_with_dtype(x, num_classes=num_global_experts, dtype=x.dtype) for x in indices_s]
    gates_score = [(scores * x).sum(dim=1) for x in masks_se]

    l_loss = loss_fn(scores, topk_indices) if loss_fn is not None else None
    if batch_prioritized_routing:
        importance_scores = -1 * scores.max(dim=1)[0]
        compute_location = lambda x: compute_sorted_location(x, importance_scores)
    else:
        compute_location = torch_cumsum_sub_one

    locations1 = compute_location(masks_se[0])
    locations_s = [torch.sum(locations1 * masks_se[0], dim=1).to(torch.int32)]

    if top_k > 1:
        acc_base = None
        for k in range(1, top_k):
            acc_base = (torch.sum(masks_se[k - 1], dim=0, keepdim=True) if acc_base is None
                        else acc_base + torch.sum(masks_se[k - 1], dim=0, keepdim=True))
            locations2 = compute_location(masks_se[k])
            locations2 += acc_base
            locations_s.append(torch.sum(locations2 * masks_se[k], dim=1).to(torch.int32))

        if normalize_gate:
            denom_s = torch.clamp(sum(gates_score), min=torch.finfo(gates_score[0].dtype).eps)
            gates_score = [x / denom_s for x in gates_score]

    samples_per_expert = (int(scores.size(0)) + num_global_experts - 1) // num_global_experts
    capacity = top_k * int(capacity_factor * samples_per_expert)

    topk_indices = topk_indices.to(torch.int32)
    locations_s = torch.stack(locations_s, dim=1)
    gates_score = torch.stack(gates_score, dim=1)

    topk_indices = torch.where(
        locations_s < capacity, topk_indices,
        torch.scalar_tensor(-1, dtype=torch.int32, device=topk_indices.device))

    return (num_global_experts, topk_indices, gates_score), l_loss

class NaiveGate(BaseGate):
    r"""
    A naive gate implementation that defines the standard behavior of the gate
    which determines which experts the tokens are going to.
    Both the indicies and the score, or confidence, are output to the parent
    module.
    The load-balance strategies are also designed to be implemented within the
    `Gate` module.
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, **kwargs):
        super().__init__(num_expert, world_size)
        # self.gate = nn.Linear(d_model, self.tot_expert)
        fp32_gate = kwargs.get('fp32_gate', True)
        self.gate = LinearTopKGate(d_model, self.tot_expert, fp32_gate=fp32_gate)
        self.top_k = top_k
        self.normalize_gate = kwargs.get('normalize_gate', False)
        self.all2all_group = kwargs.get('all2all_group', False)
        self.enforce_router = kwargs.get('enforce_router', True)
        self.batch_prioritized_routing = kwargs.get('batch_prioritized_routing', True)

    def forward(self, inp, return_all_scores=False, ):
        r"""
        The naive implementation simply calculates the top-k of a linear layer's
        output.
        """
        logits = self.gate(inp)
        if self.training and self.gate.gate_noise > 0:
            logits_w_noise = (
                logits
                + self.gate.gate_noise
                * torch.randn_like(logits)
                / self.tot_expert
            )
        else:
            logits_w_noise = logits

        scores = F.softmax(logits_w_noise, dim=1)

        if self.tot_expert == 1 or self.enforce_router:
            _loss_fn = None
        else:
            if self.is_gshard_loss:
                _loss_fn = lambda gates, topk_ids: losses.gshard_loss(
                    gates, topk_ids
                )
            else:
                _loss_fn = lambda gates, topk_ids: losses.load_importance_loss(
                    F.softmax(logits, dim=1),
                    logits_w_noise.gather(index=topk_ids, dim=1),
                    self.tot_expert,
                    self.gate.gate_noise,
                )

        (num_global_experts, gate_top_k_idx, gate_score), l_loss = extract_critical_fmoe(
                scores,
                top_k=self.top_k,
                loss_fn=_loss_fn,
                capacity_factor=self.gate.capacity_factor if capacity_factor is None else capacity_factor,
                batch_prioritized_routing=self.batch_prioritized_routing,
                normalize_gate=self.normalize_gate,
                group=self.all2all_group,
                enforce_router=self.enforce_router,
                training=self.training
            )


        if return_all_scores:
            return gate_top_k_idx, gate_score, logits
        return gate_top_k_idx, gate_score
