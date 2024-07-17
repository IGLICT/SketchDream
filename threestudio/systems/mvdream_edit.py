import os
from dataclasses import dataclass, field

import torch

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.misc import cleanup, get_device
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import torch.nn.functional as F

import nvdiffrast.torch as dr
import trimesh
import numpy as np

from threestudio.utils.misc import C, cleanup, get_device, load_module_weights

import torchvision
from PIL import Image

# For Load ref and new geometry weights
def get_keys(d, name):
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name and k[len(name)] == '.'}
    return d_filt

@threestudio.register("mvdream-system-edit")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        
        # New config for local constrain
        ref_geometry_weights: str = None
        new_geometry_weights: str = None
        ref_geometry_type: str = ""
        ref_geometry: dict = field(default_factory=dict)
        
        mesh_box_path: str = None

    cfg: Config

    def configure(self) -> None:
        # set up geometry, material, background, renderer
        super().configure()
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.requires_grad_(False)
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.prompt_utils = self.prompt_processor()
        
        # ================================================================================
        # Load reference geometry and render network
        self.ref_geometry = threestudio.find(self.cfg.ref_geometry_type)(self.cfg.ref_geometry)
        self.ref_renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.ref_geometry,
            material=self.material,
            background=self.background,
        )
        
        # Local local mesh
        mesh = trimesh.load(self.cfg.mesh_box_path)
        self.verts = torch.tensor(mesh.vertices, dtype=torch.float32).to('cuda')#.to(self.device)
        self.faces = torch.tensor(mesh.faces, dtype=torch.float32).to('cuda')#.to(self.device)
        
        if self.cfg.ref_geometry_weights != "":
            # Load the ref geometry weights
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.ref_geometry_weights, map_location="cpu"
            )
            state_dict_geo = get_keys(state_dict, 'geometry')
            self.ref_geometry.load_state_dict(state_dict_geo, strict=False)
            state_dict_render = get_keys(state_dict, 'renderer')
            self.ref_renderer.load_state_dict(state_dict_render, strict=False)
        
        if self.cfg.new_geometry_weights != "":
            # Load the new geometry weights
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.new_geometry_weights, map_location="cpu"
            )
            state_dict_geo = get_keys(state_dict, 'geometry')
            self.geometry.load_state_dict(state_dict_geo, strict=False)
            state_dict_render = get_keys(state_dict, 'renderer')
            self.renderer.load_state_dict(state_dict_render, strict=False)
        
        self.glctx = dr.RasterizeCudaContext()

    def on_load_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                return
        guidance_state_dict = {"guidance."+k : v for (k,v) in self.guidance.state_dict().items()}
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict}
        return 

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        return 

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):   
        out = self(batch)
        loss = 0.0
        
        #================================================================
        # Calculate multi-view original constrain
        with torch.no_grad():
            out_ref = self.ref_renderer(bg_color=out['comp_rgb_bg'], **batch)
        
        vertice = self.verts
        faces = self.faces
        verts_homo = torch.cat(
            [vertice, torch.ones([vertice.shape[0], 1]).to(vertice)], dim=-1
        )
        verts_trans = torch.matmul(verts_homo, batch['mvp_mtx'].permute(0, 2, 1))
        rast, _ = dr.rasterize(self.glctx, verts_trans.float(), faces.int(), resolution=[batch['height'], batch['width']])

        # calculate the depth from ordinates
        c2w = batch['c2w']
        w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
        w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
        w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
        w2c[:, 3, 3] = 1.0
        verts_trans_new = torch.matmul(verts_homo, w2c.permute(0, 2, 1))
        depth_ = torch.norm(verts_trans_new[:,:,0:3], dim=2).unsqueeze(2)
        
        depth_render_box, _ = dr.interpolate(depth_.contiguous().float(), rast.float(), faces.int())
        
        depth_nerf = out_ref['depth']
        bk_mask = out_ref['opacity'] < 0.95
        depth_nerf[bk_mask] = 100.0
        mask_box_effective = depth_render_box < depth_nerf

        mesh_mask = depth_render_box > 0.0
        edit_mask = mask_box_effective & mesh_mask
        unedit_mask = ~edit_mask
        
        batch["edit_mask"] = edit_mask.float()
        
        loss_unedit_rgb = F.mse_loss(
            out["comp_rgb"] * unedit_mask.float(),
            out_ref["comp_rgb"] * unedit_mask.float()
        )
        self.log("train/loss_unedit_rgb", loss_unedit_rgb)
        loss += loss_unedit_rgb * self.C(self.cfg.loss.lambda_unedit_rgb)
        
        if self.C(self.cfg.loss.lambda_mask) > 0 and batch['mask'] is not None:
            opacity = out["opacity"][0:1]
            
            loss_mask = F.mse_loss(batch["mask"].float() * edit_mask.float(), opacity * edit_mask.float()) * self.cfg.loss.lambda_mask
            self.log("train/loss_mask", loss_mask)
            loss += loss_mask * self.C(self.cfg.loss.lambda_mask)

        guidance_out = self.guidance(
            out["comp_rgb"], self.prompt_utils, **batch
        )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        if self.true_global_step % 100 == 0:
            x_samples = out["comp_rgb"] * unedit_mask.float()
            x_samples = x_samples.permute(0,3,1,2)
            grid = torchvision.utils.make_grid(x_samples.cpu(), nrow=4)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            Image.fromarray(grid).save("./debug/" + str(self.true_global_step) + "_mask.png")
        
            x_samples = out["comp_rgb"]
            x_samples = x_samples.permute(0,3,1,2)
            grid = torchvision.utils.make_grid(x_samples.cpu(), nrow=4)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            Image.fromarray(grid).save("./debug/" + str(self.true_global_step) + "_image.png")
        
            x_samples = batch["control"]
            grid = torchvision.utils.make_grid(x_samples.cpu(), nrow=4)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            Image.fromarray(grid).save("./debug/" + str(self.true_global_step) + "_control.png")

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
