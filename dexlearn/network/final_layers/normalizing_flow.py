import torch
from einops import rearrange, repeat

from .nflow_util import Flow
from .mlp import BasicMLP
from dexlearn.utils.rot import proper_svd
from pytorch3d import transforms as pttf


class BiFlowRT(torch.nn.Module):
    pass


class FlowRT(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.flow_thresh = 20
        self.flow = Flow(cfg.mobius, cfg.in_feat_dim, 3)
        self.flow.mask()
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) n x")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rot = hand_rot[:, -1]
        grasp_trans = hand_trans[:, -1]
        loss_nll = -self.flow.log_prob(grasp_rot, grasp_trans, global_feature)
        loss_nll[loss_nll > self.flow_thresh] = self.flow_thresh
        result_dict["loss_nll"] = loss_nll.mean()

        return result_dict

    def sample(self, global_feature, sample_num):
        grasp_rot, grasp_trans, log_prob = self.flow.sample_and_log_prob(sample_num, global_feature, allow_fail=False)
        final_trans = grasp_trans.unsqueeze(-2)
        final_quat = pttf.matrix_to_quaternion(grasp_rot.unsqueeze(-3))
        robot_pose = torch.cat([final_trans, final_quat], dim=-1)
        return robot_pose, log_prob


class FlowRT_MLPRTJ(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.flow_thresh = 20
        self.flow = Flow(cfg.mobius, cfg.in_feat_dim, 3)
        self.flow.mask()
        self.joint_mlp = BasicMLP(cfg.in_feat_dim + 9 + 3, (cfg.joint_num + 12) * cfg.traj_length - 12)
        self.joint_loss = torch.nn.SmoothL1Loss(reduction="none")
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) n x")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        hand_joint = rearrange(data["hand_joint"], "b t n x -> (b t) n x")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rot = hand_rot[:, -1]
        grasp_trans = hand_trans[:, -1]
        loss_nll = -self.flow.log_prob(grasp_rot, grasp_trans, global_feature)
        loss_nll[loss_nll > self.flow_thresh] = self.flow_thresh
        result_dict["loss_nll"] = loss_nll.mean()

        # use MLP to predict other things
        in_mlp_feat = torch.cat(
            [global_feature, rearrange(grasp_rot, "b x y -> b (x y)"), grasp_trans],
            dim=-1,
        )
        out_info = self.joint_mlp(in_mlp_feat)
        gt_info = torch.cat(
            [
                rearrange(hand_trans[:, :-1], "b n x -> b (n x)"),
                rearrange(hand_rot[:, :-1], "b n x y -> b (n x y)"),
                rearrange(hand_joint, "b n x -> b (n x)"),
            ],
            dim=-1,
        )
        loss_others = self.joint_loss(out_info, gt_info)
        pose_num = self.cfg.traj_length - 1
        result_dict["loss_trans"] = loss_others[:, : pose_num * 3].mean()
        result_dict["loss_rot"] = loss_others[:, pose_num * 3 : pose_num * 12].mean()
        result_dict["loss_joint"] = loss_others[:, pose_num * 12 :].mean()
        with torch.no_grad():
            result_dict["abs_dis_joint"] = (out_info - gt_info)[pose_num * 12 :].abs().mean()

        return result_dict

    def sample(self, global_feature, sample_num):
        grasp_rot, grasp_trans, log_prob = self.flow.sample_and_log_prob(sample_num, global_feature, allow_fail=False)

        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        in_mlp_feat = torch.cat(
            [
                global_feature,
                rearrange(grasp_rot, "b n x y -> (b n) (x y)"),
                rearrange(grasp_trans, "b n c -> (b n) c"),
            ],
            dim=-1,
        )
        pred_info = self.joint_mlp(in_mlp_feat)
        pose_num = self.cfg.traj_length - 1
        hand_trans = rearrange(
            pred_info[:, : pose_num * 3],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )
        hand_rot = rearrange(
            pred_info[:, pose_num * 3 : pose_num * 12],
            "(b t) (n x y) -> b t n x y",
            t=sample_num,
            n=pose_num,
            x=3,
        )

        hand_joint = rearrange(
            pred_info[:, pose_num * 12 :],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num + 1,
        )

        final_trans = torch.cat([hand_trans, grasp_trans.unsqueeze(-2)], dim=-2)
        final_quat = pttf.matrix_to_quaternion(
            torch.cat(
                [
                    proper_svd(hand_rot.reshape(-1, 3, 3)).reshape_as(hand_rot),
                    grasp_rot.unsqueeze(-3),
                ],
                dim=-3,
            ),
        )
        robot_pose = torch.cat([final_trans, final_quat, hand_joint], dim=-1)
        return robot_pose, log_prob


class FlowRT_MLPRTJ_woMF(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.flow_thresh = 20
        self.flow = Flow(cfg.mobius, cfg.in_feat_dim, 12)
        self.flow.mask()
        self.joint_mlp = BasicMLP(cfg.in_feat_dim + 9 + 3, (cfg.joint_num + 12) * cfg.traj_length - 12)
        self.joint_loss = torch.nn.SmoothL1Loss(reduction="none")
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) n x")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        hand_joint = rearrange(data["hand_joint"], "b t n x -> (b t) n x")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rot = hand_rot[:, -1]
        other_info = torch.cat([rearrange(grasp_rot, "b x y -> b (x y)"), hand_trans[:, -1]], dim=-1)
        loss_nll = -self.flow.log_prob(grasp_rot, other_info, global_feature)
        loss_nll[loss_nll > self.flow_thresh] = self.flow_thresh
        result_dict["loss_nll"] = loss_nll.mean()

        # use MLP to predict other things
        in_mlp_feat = torch.cat([global_feature, other_info], dim=-1)
        out_info = self.joint_mlp(in_mlp_feat)
        gt_info = torch.cat(
            [
                rearrange(hand_trans[:, :-1], "b n x -> b (n x)"),
                rearrange(hand_rot[:, :-1], "b n x y -> b (n x y)"),
                rearrange(hand_joint, "b n x -> b (n x)"),
            ],
            dim=-1,
        )
        loss_others = self.joint_loss(out_info, gt_info)
        pose_num = self.cfg.traj_length - 1
        result_dict["loss_trans"] = loss_others[:, : pose_num * 3].mean()
        result_dict["loss_rot"] = loss_others[:, pose_num * 3 : pose_num * 12].mean()
        result_dict["loss_joint"] = loss_others[:, pose_num * 12 :].mean()
        with torch.no_grad():
            result_dict["abs_dis_joint"] = (out_info - gt_info)[pose_num * 12 :].abs().mean()

        return result_dict

    def sample(self, global_feature, sample_num):
        _, grasp_rot_trans, log_prob = self.flow.sample_and_log_prob(sample_num, global_feature, allow_fail=False)
        grasp_rot = rearrange(
            proper_svd(rearrange(grasp_rot_trans[..., :9], "b n (x y) -> (b n) x y", x=3)),
            "(b n) x y -> b n x y",
            n=sample_num,
        )
        grasp_trans = grasp_rot_trans[..., 9:]

        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        in_mlp_feat = torch.cat(
            [
                global_feature,
                rearrange(grasp_rot, "b n x y -> (b n) (x y)"),
                rearrange(grasp_trans, "b n c -> (b n) c"),
            ],
            dim=-1,
        )
        pred_info = self.joint_mlp(in_mlp_feat)
        pose_num = self.cfg.traj_length - 1
        hand_trans = rearrange(
            pred_info[:, : pose_num * 3],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )
        hand_rot = rearrange(
            pred_info[:, pose_num * 3 : pose_num * 12],
            "(b t) (n x y) -> b t n x y",
            t=sample_num,
            n=pose_num,
            x=3,
        )

        hand_joint = rearrange(
            pred_info[:, pose_num * 12 :],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num + 1,
        )

        final_trans = torch.cat([hand_trans, grasp_trans.unsqueeze(-2)], dim=-2)
        final_quat = pttf.matrix_to_quaternion(
            torch.cat(
                [
                    proper_svd(hand_rot.reshape(-1, 3, 3)).reshape_as(hand_rot),
                    grasp_rot.unsqueeze(-3),
                ],
                dim=-3,
            ),
        )
        robot_pose = torch.cat([final_trans, final_quat, hand_joint], dim=-1)
        return robot_pose, log_prob


class FlowRTJ(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.flow_thresh = 20
        self.flow = Flow(cfg.mobius, cfg.in_feat_dim, (cfg.joint_num + 12) * cfg.traj_length - 9)
        self.flow.mask()
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) (n x)")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        hand_joint = rearrange(data["hand_joint"], "b t n x -> (b t) (n x)")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rot = hand_rot[:, -1]
        other_info = torch.cat(
            [
                rearrange(hand_rot[:, :-1], "b n x y -> b (n x y)"),
                hand_trans,
                hand_joint,
            ],
            dim=-1,
        )
        loss_nll = -self.flow.log_prob(grasp_rot, other_info, global_feature)
        loss_nll[loss_nll > self.flow_thresh] = self.flow_thresh
        result_dict["loss_nll"] = loss_nll.mean()

        return result_dict

    def sample(self, global_feature, sample_num):
        grasp_rot, pred_info, log_prob = self.flow.sample_and_log_prob(sample_num, global_feature, allow_fail=False)
        pose_num = self.cfg.traj_length
        other_rot = rearrange(
            pred_info[..., : (pose_num - 1) * 9],
            "b t (n x y) -> b t n x y",
            n=pose_num - 1,
            x=3,
        )
        hand_trans = rearrange(
            pred_info[..., (pose_num - 1) * 9 : pose_num * 12 - 9],
            "b t (n c) -> b t n c",
            n=pose_num,
        )

        hand_joint = rearrange(
            pred_info[..., pose_num * 12 - 9 :],
            "b t (n c) -> b t n c",
            n=pose_num,
        )

        final_quat = pttf.matrix_to_quaternion(
            torch.cat(
                [
                    proper_svd(other_rot.reshape(-1, 3, 3)).reshape_as(other_rot),
                    grasp_rot.unsqueeze(-3),
                ],
                dim=-3,
            ),
        )
        robot_pose = torch.cat([hand_trans, final_quat, hand_joint], dim=-1)
        return robot_pose, log_prob


class FlowRTJ_woMF(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.flow_thresh = 20
        self.flow = Flow(cfg.mobius, cfg.in_feat_dim, (cfg.joint_num + 12) * cfg.traj_length)
        self.flow.mask()
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) (n x)")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        hand_joint = rearrange(data["hand_joint"], "b t n x -> (b t) (n x)")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rot = hand_rot[:, -1]
        other_info = torch.cat(
            [
                rearrange(hand_rot, "b n x y -> b (n x y)"),
                hand_trans,
                hand_joint,
            ],
            dim=-1,
        )
        loss_nll = -self.flow.log_prob(grasp_rot, other_info, global_feature)
        loss_nll[loss_nll > self.flow_thresh] = self.flow_thresh
        result_dict["loss_nll"] = loss_nll.mean()

        return result_dict

    def sample(self, global_feature, sample_num):
        _, pred_info, log_prob = self.flow.sample_and_log_prob(sample_num, global_feature, allow_fail=False)
        pose_num = self.cfg.traj_length
        hand_rot = rearrange(
            pred_info[..., : pose_num * 9],
            "b t (n x y) -> b t n x y",
            n=pose_num,
            x=3,
        )
        hand_trans = rearrange(
            pred_info[..., pose_num * 9 : pose_num * 12],
            "b t (n c) -> b t n c",
            n=pose_num,
        )
        hand_joint = rearrange(
            pred_info[..., pose_num * 12 :],
            "b t (n c) -> b t n c",
            n=pose_num,
        )

        final_quat = pttf.matrix_to_quaternion(proper_svd(hand_rot.reshape(-1, 3, 3)).reshape_as(hand_rot))
        robot_pose = torch.cat([hand_trans, final_quat, hand_joint], dim=-1)
        return robot_pose, log_prob


class FlowRTJ_MLPRTJ(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.flow_thresh = 20
        self.flow = Flow(cfg.mobius, cfg.in_feat_dim, 3 + cfg.joint_num)
        self.flow.mask()
        self.joint_mlp = BasicMLP(
            cfg.in_feat_dim + cfg.joint_num + 12,
            (cfg.joint_num + 12) * (cfg.traj_length - 1),
        )
        self.joint_loss = torch.nn.SmoothL1Loss(reduction="none")
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) n x")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        hand_joint = rearrange(data["hand_joint"], "b t n x -> (b t) n x")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rot = hand_rot[:, -1]
        grasp_trans_joint = torch.cat([hand_trans[:, -1], hand_joint[:, -1]], dim=-1)
        loss_nll = -self.flow.log_prob(grasp_rot, grasp_trans_joint, global_feature)
        loss_nll[loss_nll > self.flow_thresh] = self.flow_thresh
        result_dict["loss_nll"] = loss_nll.mean()

        # use MLP to predict other things
        in_mlp_feat = torch.cat(
            [
                global_feature,
                rearrange(grasp_rot, "b x y -> b (x y)"),
                grasp_trans_joint,
            ],
            dim=-1,
        )
        out_info = self.joint_mlp(in_mlp_feat)
        gt_info = torch.cat(
            [
                rearrange(hand_trans[:, :-1], "b n x -> b (n x)"),
                rearrange(hand_rot[:, :-1], "b n x y -> b (n x y)"),
                rearrange(hand_joint[:, :-1], "b n x -> b (n x)"),
            ],
            dim=-1,
        )
        loss_others = self.joint_loss(out_info, gt_info)
        pose_num = self.cfg.traj_length - 1
        result_dict["loss_trans"] = loss_others[:, : pose_num * 3].mean()
        result_dict["loss_rot"] = loss_others[:, pose_num * 3 : pose_num * 12].mean()
        result_dict["loss_joint"] = loss_others[:, pose_num * 12 :].mean()
        with torch.no_grad():
            result_dict["abs_dis_joint"] = (out_info - gt_info)[pose_num * 12 :].abs().mean()

        return result_dict

    def sample(self, global_feature, sample_num):
        grasp_rot, grasp_trans_joint, log_prob = self.flow.sample_and_log_prob(
            sample_num, global_feature, allow_fail=False
        )

        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        in_mlp_feat = torch.cat(
            [
                global_feature,
                rearrange(grasp_rot, "b n x y -> (b n) (x y)"),
                rearrange(grasp_trans_joint, "b n c -> (b n) c"),
            ],
            dim=-1,
        )
        pred_info = self.joint_mlp(in_mlp_feat)
        pose_num = self.cfg.traj_length - 1
        hand_trans = rearrange(
            pred_info[:, : pose_num * 3],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )
        hand_rot = rearrange(
            pred_info[:, pose_num * 3 : pose_num * 12],
            "(b t) (n x y) -> b t n x y",
            t=sample_num,
            n=pose_num,
            x=3,
        )

        hand_joint = rearrange(
            pred_info[:, pose_num * 12 :],
            "(b t) (n c) -> b t n c",
            t=sample_num,
            n=pose_num,
        )

        final_trans = torch.cat([hand_trans, grasp_trans_joint[..., :3].unsqueeze(-2)], dim=-2)
        final_joint = torch.cat([hand_joint, grasp_trans_joint[..., 3:].unsqueeze(-2)], dim=-2)
        final_quat = pttf.matrix_to_quaternion(
            torch.cat(
                [
                    proper_svd(hand_rot.reshape(-1, 3, 3)).reshape_as(hand_rot),
                    grasp_rot.unsqueeze(-3),
                ],
                dim=-3,
            ),
        )
        robot_pose = torch.cat([final_trans, final_quat, final_joint], dim=-1)
        return robot_pose, log_prob


class FlowRT_MLPJ(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.flow_thresh = 20
        self.flow = Flow(cfg.mobius, cfg.in_feat_dim, 12 * cfg.traj_length - 9)
        self.flow.mask()
        self.joint_mlp = BasicMLP(cfg.in_feat_dim + 12 * cfg.traj_length, cfg.joint_num * cfg.traj_length)
        self.joint_loss = torch.nn.SmoothL1Loss(reduction="none")
        return

    def forward(self, data, global_feature):
        result_dict = {}

        # Deal with multiple grasps for one vision input
        batch_num, sample_num, pose_num, _ = data["hand_trans"].shape
        hand_trans = rearrange(data["hand_trans"], "b t n x -> (b t) n x")
        hand_rot = rearrange(data["hand_rot"], "b t n x y -> (b t) n x y")
        hand_joint = rearrange(data["hand_joint"], "b t n x -> (b t) n x")
        global_feature = repeat(global_feature, "b c -> (b t) c", t=sample_num)

        # Use Flow to predict grasp rot and trans
        grasp_rot = hand_rot[:, -1]
        other_rt = torch.cat(
            [
                rearrange(hand_rot[:, :-1], "b n x y -> b (n x y)"),
                rearrange(hand_trans, "b n x -> b (n x)"),
            ],
            dim=-1,
        )
        loss_nll = -self.flow.log_prob(grasp_rot, other_rt, global_feature)
        loss_nll[loss_nll > self.flow_thresh] = self.flow_thresh
        result_dict["loss_nll"] = loss_nll.mean()

        # use MLP to predict other things
        in_mlp_feat = torch.cat(
            [global_feature, rearrange(grasp_rot, "b x y -> b (x y)"), other_rt],
            dim=-1,
        )
        out_info = self.joint_mlp(in_mlp_feat)
        gt_info = rearrange(hand_joint, "b n x -> b (n x)")

        result_dict["loss_joint"] = self.joint_loss(out_info, gt_info).mean()
        with torch.no_grad():
            result_dict["abs_dis_joint"] = (out_info - gt_info).abs().mean()

        return result_dict

    def sample(self, global_feature, sample_num):
        grasp_rot, other_rt, log_prob = self.flow.sample_and_log_prob(sample_num, global_feature, allow_fail=False)

        global_feature = repeat(global_feature, "b c -> (b n) c", n=sample_num)

        pose_num = self.cfg.traj_length
        hand_trans = rearrange(other_rt[..., (pose_num - 1) * 9 :], "b t (n c) -> b t n c", n=pose_num)
        hand_rot = rearrange(
            other_rt[..., : (pose_num - 1) * 9],
            "b t (n x y) -> b t n x y",
            n=pose_num - 1,
            x=3,
        )
        hand_rot = proper_svd(hand_rot.reshape(-1, 3, 3)).reshape_as(hand_rot)
        in_mlp_feat = torch.cat(
            [
                global_feature,
                rearrange(grasp_rot, "b t x y -> (b t) (x y)"),
                rearrange(hand_rot, "b t n x y -> (b t) (n x y)"),
                rearrange(hand_trans, "b t n x -> (b t) (n x)"),
            ],
            dim=-1,
        )
        pred_info = self.joint_mlp(in_mlp_feat)
        hand_joint = rearrange(pred_info, "(b t) (n c) -> b t n c", t=sample_num, n=pose_num)

        final_quat = pttf.matrix_to_quaternion(torch.cat([hand_rot, grasp_rot.unsqueeze(-3)], dim=-3))
        robot_pose = torch.cat([hand_trans, final_quat, hand_joint], dim=-1)
        return robot_pose, log_prob
