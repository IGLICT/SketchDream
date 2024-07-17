import os
from dataclasses import dataclass, field

import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *

import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr
import trimesh
import numpy as np

from threestudio.utils.misc import C, cleanup, get_device, load_module_weights

import torchvision
from PIL import Image

import random
import math
import cubvh

# For Load ref and new geometry weights
def get_keys(d, name):
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name and k[len(name)] == '.'}
    return d_filt

@threestudio.register("mvdream-system-edit-lucid-refine-local")
class MVDreamSystem(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        visualize_samples: bool = False
        
        # LucidDreamer guidance module
        lucid_guidance_type: str = ""
        lucid_guidance: dict = field(default_factory=dict)
        
        lucid_prompt_processor_type: str = ""
        lucid_prompt_processor: dict = field(default_factory=dict)
        
        # 2D LucidDream probablity
        Diffusion_2D_prob: dict = field(default_factory=dict)
        
        # New config for local constrain
        ref_geometry_weights: str = ""
        new_geometry_weights: str = ""
        ref_geometry_type: str = ""
        ref_geometry: dict = field(default_factory=dict)
        
        # mesh_box_path: str = None
        mesh_box_path: List[str] = field(default_factory=lambda: [])
        mesh_box_path_render: str = ""
        
        local_camera_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        local_camera_distance: float = 1.0
        
        local_prob: float = 0.5

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
        # Load lucidDreamer
        # Load 2D StableDiffusion for lucidDreamer
        self.guidance_2d = threestudio.find(self.cfg.lucid_guidance_type)(self.cfg.lucid_guidance)
        self.prompt_processor_2d = threestudio.find(self.cfg.lucid_prompt_processor_type)(
            self.cfg.lucid_prompt_processor
        )
        
        self.Diffusion_2D_prob = self.cfg.Diffusion_2D_prob
        self.local_prob = self.cfg.local_prob
        
        # ================================================================================
        # Load reference geometry and render network
        self.ref_geometry = threestudio.find(self.cfg.ref_geometry_type)(self.cfg.ref_geometry)
        self.ref_renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.ref_geometry,
            material=self.material,
            background=self.background,
        )
        
        self.mesh_verts = []
        self.mesh_faces = []
        self.mesh_color = []
        for mesh_path in self.cfg.mesh_box_path:
            mesh = trimesh.load(mesh_path)
            self.mesh_verts.append(torch.tensor(mesh.vertices, dtype=torch.float32).to('cuda')) #.to(self.device)
            self.mesh_faces.append(torch.tensor(mesh.faces, dtype=torch.float32).to('cuda')) #.to(self.device)
            # Get the vertex color
            vertex_colors = mesh.visual.vertex_colors
            vertex_colors = 255.0 - vertex_colors
            self.mesh_color.append(torch.tensor(vertex_colors, dtype=torch.float32).to('cuda'))
        
        # # Local local mesh
        # mesh_box = trimesh.load(self.cfg.mesh_box_path_render)
        # self.BVH_mesh = cubvh.cuBVH(mesh_box.vertices, mesh_box.faces) # build with numpy.ndarray/torch.Tensor
        
        # ======================================================================
        # local render parameters
        ball_center = torch.tensor([self.cfg.local_camera_position]).to('cuda')
        self.camera_distance_local = self.cfg.local_camera_distance
        
        self.ball_center_homo = torch.cat(
            [ball_center, torch.ones([ball_center.shape[0], 1]).to('cuda')], dim=-1
        )
        resolution = 256
        uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device='cpu'), torch.arange(resolution, dtype=torch.float32, device='cpu'))) * (1./resolution) + (0.5/resolution)
        uv = uv * 2.0 - 1.0
        uv = uv.permute(1,2,0)
        uv = torch.unsqueeze(uv, dim=0)
        sketch_grid_256 = uv[:,:,:,[1,0]]
        self.sketch_grid_256 = sketch_grid_256.to('cuda')
        
        resolution = 64
        uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device='cpu'), torch.arange(resolution, dtype=torch.float32, device='cpu'))) * (1./resolution) + (0.5/resolution)
        uv = uv * 2.0 - 1.0
        uv = uv.permute(1,2,0)
        uv = torch.unsqueeze(uv, dim=0)
        sketch_grid_64 = uv[:,:,:,[1,0]]
        self.sketch_grid_64 = sketch_grid_64.to('cuda')
        # ======================================================================
        
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
        
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance_2d."):
                return
        guidance_2d_state_dict = {"guidance_2d."+k : v for (k,v) in self.guidance_2d.state_dict().items()}
        
        checkpoint['state_dict'] = {**checkpoint['state_dict'], **guidance_state_dict, **guidance_2d_state_dict}
        return

    def on_save_checkpoint(self, checkpoint):
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance."):
                checkpoint['state_dict'].pop(k)
        
        for k in list(checkpoint['state_dict'].keys()):
            if k.startswith("guidance_2d."):
                checkpoint['state_dict'].pop(k)
        
        return 

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.renderer(**batch)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        with torch.no_grad():
            out_ref = self.ref_renderer(bg_color=out['comp_rgb_bg'], **batch)
        
        loss = 0.0
        #================================================================
        # Calculate multi-view original constrain        
        binary_tensor = None
        for mesh_index in range(len(self.mesh_verts)):
            vertice = self.mesh_verts[mesh_index]
            faces = self.mesh_faces[mesh_index]
            color = self.mesh_color[mesh_index]
            
            verts_homo = torch.cat(
                [vertice, torch.ones([vertice.shape[0], 1]).to(vertice)], dim=-1
            )
            verts_trans = torch.matmul(verts_homo, batch['mvp_mtx'].permute(0, 2, 1))
            rast, _ = dr.rasterize(self.glctx, verts_trans.float(), faces.int(), resolution=[batch['height'], batch['width']])
            
            refine_mask, _ = dr.interpolate(color.contiguous().float(), rast.float(), faces.int())
            refine_mask = torch.mean(refine_mask[:,:,:,0:3], dim=3, keepdim=True)
            
            dtype = out['comp_rgb_bg'].dtype
            threshold = 200   
            if binary_tensor is None:
                binary_tensor = refine_mask > threshold
            else:
                binary_tensor = binary_tensor | (refine_mask > threshold)
            
        binary_tensor = torch.where(binary_tensor, torch.tensor(1, dtype=dtype).to(refine_mask.device), torch.tensor(0, dtype=dtype).to(refine_mask.device))

        # ==============================================================
        if random.random() < self.local_prob:
            c2w = batch['c2w']
            w2c: Float[Tensor, "B 4 4"] = torch.zeros(c2w.shape[0], 4, 4).to(c2w)
            w2c[:, :3, :3] = c2w[:, :3, :3].permute(0, 2, 1)
            w2c[:, :3, 3:] = -c2w[:, :3, :3].permute(0, 2, 1) @ c2w[:, :3, 3:]
            w2c[:, 3, 3] = 1.0
            
            verts_trans = torch.matmul(self.ball_center_homo, w2c.permute(0, 2, 1))
            # camera_distance_range: the distance between the camera and center
            camera_distance_range = abs(verts_trans[:,:,2])
            # calculate the scale based on camera distance
            sketch_scale = self.camera_distance_local / camera_distance_range
            
            if batch['height'] == 64:
                sketch_grid = self.sketch_grid_64.repeat(5,1,1,1) * sketch_scale.reshape(5,1,1,1)
                sketch_grid_256 = self.sketch_grid_256.repeat(5,1,1,1) * sketch_scale.reshape(5,1,1,1)
            else:
                sketch_grid = self.sketch_grid_256.repeat(5,1,1,1) * sketch_scale.reshape(5,1,1,1)

            local_fov = batch['fovy_deg'] * math.pi / 180
            sketch_fovy = torch.tan(0.5 * local_fov)
        
            delta_right = verts_trans[:,0,0]
            delta_pixel_right = delta_right / (camera_distance_range[:,0] * sketch_fovy)
            sketch_grid[:,:,:,0] = sketch_grid[:,:,:,0] + delta_pixel_right.reshape(5,1,1)
        
            delta_up = verts_trans[:,0,1]
            delta_pixel_up = -delta_up / (camera_distance_range[:,0] * sketch_fovy)
            sketch_grid[:,:,:,1] = sketch_grid[:,:,:,1] + delta_pixel_up.reshape(5,1,1)
        
            out["comp_rgb"] = F.grid_sample(out["comp_rgb"].permute(0,3,1,2), sketch_grid, mode='bilinear', padding_mode='border', align_corners=False).permute(0,2,3,1)
            out_ref["comp_rgb"] = F.grid_sample(out_ref["comp_rgb"].permute(0,3,1,2), sketch_grid, mode='bilinear', padding_mode='border', align_corners=False).permute(0,2,3,1)
            
            if batch['height'] == 64:
                sketch_grid_256[:,:,:,0] = sketch_grid_256[:,:,:,0] + delta_pixel_right.reshape(5,1,1)
                sketch_grid_256[:,:,:,1] = sketch_grid_256[:,:,:,1] + delta_pixel_up.reshape(5,1,1)
                batch["control"] = F.grid_sample(batch["control"], sketch_grid_256, mode='bilinear', padding_mode='border', align_corners=False)
            else:
                batch["control"] = F.grid_sample(batch["control"], sketch_grid, mode='bilinear', padding_mode='border', align_corners=False)
            
            kernel_size = 5
            binary_tensor = binary_tensor.permute(0,3,1,2)
            binary_tensor = F.grid_sample(binary_tensor, sketch_grid, mode='bilinear', padding_mode='border', align_corners=False) #.permute(0,2,3,1)
            dilated_mask = F.conv2d(binary_tensor, torch.ones(1, 1, kernel_size, kernel_size, dtype=dtype).to(binary_tensor.device), padding=kernel_size // 2)
            edit_mask = dilated_mask.permute(0,2,3,1)
        else:
            kernel_size = 5
            binary_tensor = binary_tensor.permute(0,3,1,2)
            dilated_mask = F.conv2d(binary_tensor, torch.ones(1, 1, kernel_size, kernel_size, dtype=dtype).to(binary_tensor.device), padding=kernel_size // 2)
            edit_mask = dilated_mask.permute(0,2,3,1)
        # ==============================================================

        edit_mask = edit_mask.clip(0,1).detach()
        unedit_mask = 1 - edit_mask
        
        batch["edit_mask"] = edit_mask.float()
        loss_unedit_rgb = F.mse_loss(
            out["comp_rgb"] * unedit_mask.float(),
            out_ref["comp_rgb"] * unedit_mask.float()
        )
        self.log("train/loss_unedit_rgb", loss_unedit_rgb)
        loss += loss_unedit_rgb * self.C(self.cfg.loss.lambda_unedit_rgb)
        
        if self.C(self.cfg.loss.lambda_mask) > 0 and batch['mask'] is not None:
            opacity = out["opacity"][0:1]
            
            loss_mask = F.mse_loss(batch["mask"].float() * edit_mask.float(), opacity * edit_mask.float()) * self.C(self.cfg.loss.lambda_mask)
            self.log("train/loss_mask", loss_mask)
            loss += loss_mask * self.C(self.cfg.loss.lambda_mask)
        
        is_2D_loss = random.random() < self.C(self.cfg.Diffusion_2D_prob)
        
        if is_2D_loss:
            batch["elevation"] = batch["elevation"][1:]
            batch["azimuth"] = batch["azimuth"][1:]
            guidance_out = self.guidance_2d(
                out["comp_rgb"][1:], out["depth"][1:], out["opacity"][1:], self.prompt_processor_2d, self.true_global_step, **batch
            )
            # print("use 2D loss")
        else:
            guidance_out = self.guidance(
                out["comp_rgb"], self.prompt_utils, **batch
            )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
        
        if self.true_global_step % 50 == 0:
            if not os.path.exists("./debug/editing/"):
                os.makedirs("./debug/editing/")
            x_samples = out["comp_rgb"] * unedit_mask.float()
            x_samples = x_samples.permute(0,3,1,2)
            grid = torchvision.utils.make_grid(x_samples.cpu(), nrow=4)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            Image.fromarray(grid).save("./debug/editing/" + str(self.true_global_step) + "_mask.png")
        
            x_samples = out["comp_rgb"]
            x_samples = x_samples.permute(0,3,1,2)
            grid = torchvision.utils.make_grid(x_samples.cpu(), nrow=4)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            Image.fromarray(grid).save("./debug/editing/" + str(self.true_global_step) + "_image.png")
        
            x_samples = batch["control"]
            grid = torchvision.utils.make_grid(x_samples.cpu(), nrow=4)
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            Image.fromarray(grid).save("./debug/editing/" + str(self.true_global_step) + "_control.png")

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
