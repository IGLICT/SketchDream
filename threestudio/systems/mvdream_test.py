import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import torch.nn.functional as F

import numpy as np
from PIL import Image
import cv2


@threestudio.register("mvdream-system-test")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()

    def on_load_checkpoint(self, checkpoint):
        # Just a debug code for local editing
        for k in list(checkpoint['state_dict'].keys()):
            if 'ref' in k:
                checkpoint['state_dict'].pop(k)
        
        return 
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)

        guidance_out = self.guidance(
            out["comp_rgb"], self.prompt_utils, **batch
        )

        loss = 0.0

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                
        if self.C(self.cfg.loss.lambda_mask) > 0 and batch['mask'] is not None:
            opacity = out["opacity"][0:1]
            
            loss_mask = F.mse_loss(batch["mask"].float(), opacity) * self.cfg.loss.lambda_mask
            self.log("train/loss_mask", loss_mask)
            loss += loss_mask * self.C(self.cfg.loss.lambda_mask)

        if self.C(self.cfg.loss.lambda_orient) > 0:
            if "normal" not in out:
                raise ValueError(
                    "Normal is required for orientation loss, no normal is found in the output."
                )
            loss_orient = (
                out["weights"].detach()
                * dot(out["normal"], out["t_dirs"]).clamp_min(0.0) ** 2
            ).sum() / (out["opacity"] > 0).sum()
            self.log("train/loss_orient", loss_orient)
            loss += loss_orient * self.C(self.cfg.loss.lambda_orient)

        if self.C(self.cfg.loss.lambda_sparsity) > 0:
            loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
            self.log("train/loss_sparsity", loss_sparsity)
            loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        if self.C(self.cfg.loss.lambda_opaque) > 0:
            opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
            loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
            self.log("train/loss_opaque", loss_opaque)
            loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)

        # z variance loss proposed in HiFA: http://arxiv.org/abs/2305.18766
        # helps reduce floaters and produce solid geometry
        if self.C(self.cfg.loss.lambda_z_variance) > 0:
            loss_z_variance = out["z_variance"][out["opacity"] > 0.5].mean()
            self.log("train/loss_z_variance", loss_z_variance)
            loss += loss_z_variance * self.C(self.cfg.loss.lambda_z_variance)

        if hasattr(self.cfg.loss, "lambda_eikonal") and self.C(self.cfg.loss.lambda_eikonal) > 0:
            loss_eikonal = (
                (torch.linalg.norm(out["sdf_grad"], ord=2, dim=-1) - 1.0) ** 2
            ).mean()
            self.log("train/loss_eikonal", loss_eikonal)
            loss += loss_eikonal * self.C(self.cfg.loss.lambda_eikonal)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        
        if batch_idx == 0:
            save_dict = {}
            save_dict['rays_d'] = batch['rays_d'].cpu().numpy()
            save_dict['rays_o'] = batch['rays_o'].cpu().numpy()
            save_dict['depth'] = out['depth'].cpu().numpy()
            path = self.get_save_path('./edit_param.npy')
            np.save(path, save_dict)
            
            img = out["comp_rgb"][0].clip(0,1)
            img = img.cpu().numpy()
            img = (img * 255).astype(np.uint8)
            path = self.get_save_path('./eval_view.png')
            Image.fromarray(img).save(path)
            
            # 修改原点的距离为z平面距离
            depth_point = batch['rays_d'] * out['depth'] + batch['rays_o']
            depth_point = depth_point.reshape(-1, 3)
            vertice = depth_point
            verts_homo = torch.cat(
                        [vertice, torch.ones([vertice.shape[0], 1]).to(vertice)], dim=-1
                    )
            c2w = batch['c2w']
            w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
            w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
            w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
            w2c[:, 3, 3] = 1.0
            verts_trans_new = torch.matmul(verts_homo, w2c.permute(0, 2, 1))
            depth_new = verts_trans_new[0,:,2]
            depth_new = abs(depth_new).reshape(1,512,512,1)
            depth_new_np = depth_new.cpu().numpy().astype('float32')
            bk = depth_new_np < 0.01
            
            depth1 = depth_new_np
            depth1[bk] = 10.0
            depth1 = cv2.resize(depth1[0], (256,256)) # [:,:,0]
            
            save_path = self.get_save_path('./edit_depth_nerf.npy')
            np.save(save_path, depth1)
        
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="validation_step",
            step=self.true_global_step,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        # self.background.env_color = torch.tensor([1.0,1.0,1.0]).to('cuda')
        # batch['bg_color'] = torch.tensor([1.0,1.0,1.0]).to('cuda')
        
        out = self(batch)

        self.save_image_grid(
            f"it{self.true_global_step}-test/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                if "comp_rgb" in out
                else []
            )
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            )
            + [
                {
                    "type": "grayscale",
                    "img": out["opacity"][0, :, :, 0],
                    "kwargs": {"cmap": None, "data_range": (0, 1)},
                },
            ],
            name="test_step",
            step=self.true_global_step,
        )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )
