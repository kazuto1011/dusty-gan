import os.path as osp

import torch
from torch.utils.cpp_extension import load

module_path = osp.dirname(__file__)
emd = load(
    name="emd",
    sources=[
        osp.join(module_path, "earth_mover_distance.cpp"),
        osp.join(module_path, "earth_mover_distance.cu"),
    ],
)


class EarthMoverDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd.approxmatch_forward(xyz1, xyz2)
        cost = emd.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd.matchcost_backward(grad_cost, xyz1, xyz2, match)
        return grad_xyz1, grad_xyz2


class EarthMoverDistance(torch.nn.Module):
    def forward(self, input1, input2, eps, iters):
        return EarthMoverDistanceFunction.apply(input1, input2)


earth_mover_distance = EarthMoverDistanceFunction.apply
