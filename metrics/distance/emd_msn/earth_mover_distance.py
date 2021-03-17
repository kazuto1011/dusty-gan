# EMD approximation module (based on auction algorithm)
# memory complexity: O(n)
# time complexity: O(n^2 * iter)
# author: Minghua Liu

# Input:
# xyz1, xyz2: [#batch, #points, 3]
# where xyz1 is the predicted point cloud and xyz2 is the ground truth point cloud
# two point clouds should have same size and be normalized to [0, 1]
# #points should be a multiple of 1024
# #batch should be no greater than 512
# eps is a parameter which balances the error rate and the speed of convergence
# iters is the number of iteration
# we only calculate gradient for xyz1

# Output:
# dist: [#batch, #points],  sqrt(dist) -> L2 distance
# assignment: [#batch, #points], index of the matched point in the ground truth point cloud
# the result is an approximation and the assignment is not guranteed to be a bijection

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
    def forward(ctx, xyz1, xyz2, eps, iters):

        batchsize, n, _ = xyz1.size()
        _, m, _ = xyz2.size()

        assert n == m
        assert xyz1.size()[0] == xyz2.size()[0]
        assert n % 1024 == 0
        assert batchsize <= 512

        xyz1 = xyz1.contiguous().float().cuda()
        xyz2 = xyz2.contiguous().float().cuda()
        dist = torch.zeros(batchsize, n, device="cuda").contiguous()
        assignment = (
            torch.zeros(batchsize, n, device="cuda", dtype=torch.int32).contiguous() - 1
        )
        assignment_inv = (
            torch.zeros(batchsize, m, device="cuda", dtype=torch.int32).contiguous() - 1
        )
        price = torch.zeros(batchsize, m, device="cuda").contiguous()
        bid = torch.zeros(batchsize, n, device="cuda", dtype=torch.int32).contiguous()
        bid_increments = torch.zeros(batchsize, n, device="cuda").contiguous()
        max_increments = torch.zeros(batchsize, m, device="cuda").contiguous()
        unass_idx = torch.zeros(
            batchsize * n, device="cuda", dtype=torch.int32
        ).contiguous()
        max_idx = torch.zeros(
            batchsize * m, device="cuda", dtype=torch.int32
        ).contiguous()
        unass_cnt = torch.zeros(512, dtype=torch.int32, device="cuda").contiguous()
        unass_cnt_sum = torch.zeros(512, dtype=torch.int32, device="cuda").contiguous()
        cnt_tmp = torch.zeros(512, dtype=torch.int32, device="cuda").contiguous()

        emd.forward(
            xyz1,
            xyz2,
            dist,
            assignment,
            price,
            assignment_inv,
            bid,
            bid_increments,
            max_increments,
            unass_idx,
            unass_cnt,
            unass_cnt_sum,
            cnt_tmp,
            max_idx,
            eps,
            iters,
        )

        ctx.save_for_backward(xyz1, xyz2, assignment)
        return dist, assignment

    @staticmethod
    def backward(ctx, graddist, gradidx):
        xyz1, xyz2, assignment = ctx.saved_tensors
        graddist = graddist.contiguous()

        gradxyz1 = torch.zeros(xyz1.size(), device="cuda").contiguous()
        gradxyz2 = torch.zeros(xyz2.size(), device="cuda").contiguous()

        emd.backward(xyz1, xyz2, gradxyz1, graddist, assignment)
        return gradxyz1, gradxyz2, None, None


class EarthMoverDistance(torch.nn.Module):
    def forward(self, input1, input2, eps, iters):
        return EarthMoverDistanceFunction.apply(input1, input2, eps, iters)


earth_mover_distance = EarthMoverDistanceFunction.apply
