import math
import random
from dataclasses import dataclass

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.utils.config import parse_structured
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *

# # Multi-process, more quickly, with small bug now. Sometime generate dead thread
# from threestudio.data.uncond_multi import (
#     RandomCameraDataModuleConfig,
#     RandomCameraIterableDataset,
# )
# from threestudio.data.uncond_multi import lock

from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraIterableDataset,
)
import einops


@dataclass
class RandomMultiviewCameraDataModuleConfig(RandomCameraDataModuleConfig):
    relative_radius: bool = False
    n_view: int = 1
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    azimuth_input: float = 0.0
    sketch_path: str = ""
    sketch_fovy: float = 40
    
    mask_path: str = ""
    depth_path: str = ""

from threestudio.data.dataset_objaverse_sketch_np import Warper, sketch_warp_new

class RandomMultiviewCameraIterableDataset(RandomCameraIterableDataset):
    # This should render n_view + 1 images
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_range = self.cfg.zoom_range
        
        # Read input source image
        source = cv2.imread(self.cfg.sketch_path)
        source = source[:,:,0:1]
        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0
        self.source = np.expand_dims(source, axis=0)
        
        if self.cfg.mask_path != "":
            rgba = cv2.cvtColor(
            cv2.imread(self.cfg.mask_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA
            )
            rgba = rgba.astype(np.float32) / 255.0

            mask: Float[Tensor, "1 H W 1"] = (
                torch.from_numpy(rgba[..., 3:] > 0.5).unsqueeze(0)
            )
            self.mask = einops.rearrange(mask, 'f h w c -> f c h w').float()
        else:
            self.mask = None
        
        white_mask = torch.ones(self.source.shape)
        self.white_mask = einops.rearrange(white_mask, 'f h w c -> f c h w')
        
        # Read Depth map
        depth = np.load(self.cfg.depth_path)
        # Warp mask
        mask_img = depth != 1
        mask_img = np.ones(mask_img.shape) * 255 * mask_img
        kernel = np.ones((3,3),np.uint8)
        mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
        mask_img = cv2.erode(mask_img, kernel,iterations = 1).reshape(1, 1, 256,256)
        self.mask_img = torch.tensor(mask_img).to(torch.float32)
        
        self.depth = torch.tensor(depth).to(torch.float32)
        
        self.sketch_elevation = self.cfg.eval_elevation_deg
        sketch_fovy = torch.tensor(self.cfg.sketch_fovy)
        self.sketch_fovy = sketch_fovy * math.pi / 180
        resolution = 256
        uv = torch.stack(torch.meshgrid(torch.arange(resolution, dtype=torch.float32, device='cpu'), torch.arange(resolution, dtype=torch.float32, device='cpu'))) * (1./resolution) + (0.5/resolution)
        uv = uv * 2.0 - 1.0
        uv = uv.permute(1,2,0)
        uv = torch.unsqueeze(uv, dim=0)
        self.sketch_grid = uv[:,:,:,[1,0]]
        
        self.warper = Warper()

    def collate(self, batch) -> Dict[str, Any]:
        assert self.batch_size % (self.cfg.n_view + 1) == 0, f"batch_size ({self.batch_size}) must be dividable by n_view ({(self.cfg.n_view + 1)})!"
        real_batch_size = self.batch_size // (self.cfg.n_view + 1)
        
        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        repeat_num = self.cfg.n_view + 1
        
        # Sample elevation by two ways
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            # concat with sketch elevation
            elevation_deg = elevation_deg.reshape(real_batch_size, self.cfg.n_view)
            elevation_input_deg = torch.tensor([self.cfg.eval_elevation_deg]).reshape(1, 1).repeat(real_batch_size, 1)
            elevation_deg = torch.cat([elevation_input_deg, elevation_deg], dim=1).reshape(-1)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            # concat with sketch elevation
            elevation = elevation.reshape(real_batch_size, self.cfg.n_view)
            elevation_input_deg = torch.tensor([self.cfg.eval_elevation_deg]).reshape(1, 1).repeat(real_batch_size, 1)
            elevation_input = elevation_input_deg * math.pi / 180
            elevation = torch.cat([elevation_input, elevation], dim=1).reshape(-1)
            # change into degree format
            elevation_deg = elevation / math.pi * 180.0
        
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave((repeat_num), dim=0)
        fovy = fovy_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave((repeat_num), dim=0)
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave((repeat_num), dim=0)
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom

        # The start azimuth should within [sketch_azimuth - 45, sketch_azimuth + 45]
        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        azimuth_deg = (torch.arange(self.cfg.n_view).reshape(1,-1)).reshape(-1) / self.cfg.n_view * 360 + \
            torch.rand(real_batch_size).reshape(-1,1) * (self.azimuth_range[1] - self.azimuth_range[0]) + self.azimuth_range[0]
        
        azimuth_deg = azimuth_deg + self.cfg.azimuth_input
        # concat sample azimuth with sketch azimuth
        azimuth_deg = azimuth_deg.reshape(real_batch_size, self.cfg.n_view)
        azimuth_input = torch.tensor([self.cfg.azimuth_input]).reshape(1, 1).repeat(real_batch_size, 1)
        azimuth_deg = torch.cat([azimuth_input, azimuth_deg], dim=1).reshape(-1)
        
        # base on the fovy，change the scale of sketch and other attributes
        sketch_scale = torch.tan(0.5 * fovy[0]) / torch.tan(0.5 * self.sketch_fovy)
        sketch_grid = self.sketch_grid * sketch_scale
        # get the scale input
        source = torch.tensor(self.source)
        source = einops.rearrange(source, 'f h w c -> f c h w').clone()
        scale_input = torch.cat([source, self.white_mask, self.depth.reshape(1,1,256,256), self.mask, self.mask_img], dim=1)
        # scale the sketch and other attributes
        scale_output = F.grid_sample(scale_input, sketch_grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        
        # generate input source image
        output = scale_output[:,0:1,:,:]
        mask_sample = scale_output[:,1:2,:,:] < 0.5
        source_new = output * ~mask_sample + self.white_mask * mask_sample
        # generate input depth image
        depth = scale_output[:,2:3,:,:]
        depth = depth * ~mask_sample + self.white_mask * mask_sample
        depth = depth.numpy()[0,0,:,:]
        # generate depth mask image
        mask_img = scale_output[:,4:5,:,:].numpy().reshape(256,256,1)
        
        # warp the input sketch
        source_list = []
        sketch_warper_input = source_new[0].permute(1,2,0).numpy() * 255.0
        for i in range(5):
            if i == 0:
                # input view: mask the sketch with depth
                source_warp = sketch_warper_input * (mask_img > 125.0)
                source_list.append(source_warp.reshape(256,256,1))
            elif i == 1:
                # warp into novel view
                source_warp = sketch_warp_new(self.warper, mask_img, sketch_warper_input, depth, \
                   np.deg2rad(self.cfg.eval_elevation_deg), np.deg2rad(elevation_deg[i].item()), np.deg2rad(self.cfg.azimuth_input), np.deg2rad(azimuth_deg[i].item()))
                source_list.append(source_warp.reshape(256,256,1))
            else:
                source_list.append(np.zeros((256,256,1)))
        # get the control sketch condition
        source_new = np.stack(source_list, axis=0) / 255.0
        source_new = torch.tensor(source_new).permute(0,3,1,2).to(torch.float32)
        # concate with angle maps
        target_view_channels = torch.ones([self.batch_size, 1, 256, 256]) * ((azimuth_deg.reshape(-1,1,1,1) - self.cfg.azimuth_input) % 360.0) / 360.0
        elevation_view_channels = torch.ones([self.batch_size, 1, 256, 256]) * ((elevation_deg.reshape(-1,1,1,1) - self.cfg.eval_elevation_deg) % 360.0) / 360.0
        control = torch.cat([source_new, target_view_channels, elevation_view_channels], dim=1)
        
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(real_batch_size * repeat_num, 1)

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(repeat_num, dim=0)

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + torch.randn(real_batch_size, 3).repeat_interleave(repeat_num, dim=0) * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(repeat_num, dim=0)  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(repeat_num, dim=0)  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0
        
        # # Multi-process, more quickly, with small bug now
        # with lock:
        #     height = self.height.value
        #     width = self.width.value
        #     directions_unit_focal = self.directions_unit_focal[0]
        
        # Single process
        height = self.height
        width = self.width
        directions_unit_focal = self.directions_unit_focal

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(real_batch_size * repeat_num, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, width / height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        # get the input mask
        mask = scale_output[:,3:4,:,:]
        mask = F.interpolate(mask, size=[width, height], mode="bilinear")
        mask = einops.rearrange(mask, 'f c h w -> f h w c').clone()

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": height,
            "width": width,
            "fovy": fovy_deg,
            "control": control,
            "mask": mask,
            "fovy_deg": fovy_deg,
            "source": output,
        }

class RandomCameraDataset_sketch(Dataset):
    def __init__(self, cfg: Any, split: str) -> None:
        super().__init__()
        self.cfg: RandomCameraDataModuleConfig = cfg
        self.split = split

        if split == "val":
            self.n_views = self.cfg.n_val_views
        else:
            self.n_views = self.cfg.n_test_views

        azimuth_deg: Float[Tensor, "B"]
        if self.split == "val":
            # make sure the first and last view are not the same
            azimuth_deg = torch.linspace(0, 360.0, self.n_views + 1)[: self.n_views]
            azimuth_deg = torch.cat((torch.tensor([self.cfg.azimuth_input]), azimuth_deg), dim=0)
            self.n_views += 1
        else:
            azimuth_deg = torch.linspace(0 + self.cfg.azimuth_input, 360.0 + self.cfg.azimuth_input, self.n_views)
        
        elevation_deg: Float[Tensor, "B"] = torch.full_like(
            azimuth_deg, self.cfg.eval_elevation_deg
        )
        camera_distances: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_camera_distance
        )

        elevation = elevation_deg * math.pi / 180
        azimuth = azimuth_deg * math.pi / 180

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.cfg.eval_batch_size, 1)

        fovy_deg: Float[Tensor, "B"] = torch.full_like(
            elevation_deg, self.cfg.eval_fovy_deg
        )
        fovy = fovy_deg * math.pi / 180
        light_positions: Float[Tensor, "B 3"] = camera_positions

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = (
            0.5 * self.cfg.eval_height / torch.tan(0.5 * fovy)
        )
        directions_unit_focal = get_ray_directions(
            H=self.cfg.eval_height, W=self.cfg.eval_width, focal=1.0
        )
        
        directions: Float[Tensor, "B H W 3"] = directions_unit_focal[
            None, :, :, :
        ].repeat(self.n_views, 1, 1, 1)
        
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.cfg.eval_width / self.cfg.eval_height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        self.rays_o, self.rays_d = rays_o, rays_d
        self.mvp_mtx = mvp_mtx
        self.c2w = c2w
        self.camera_positions = camera_positions
        self.light_positions = light_positions
        self.elevation, self.azimuth = elevation, azimuth
        self.elevation_deg, self.azimuth_deg = elevation_deg, azimuth_deg
        self.camera_distances = camera_distances

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        return {
            "index": index,
            "rays_o": self.rays_o[index],
            "rays_d": self.rays_d[index],
            "mvp_mtx": self.mvp_mtx[index],
            "c2w": self.c2w[index],
            "camera_positions": self.camera_positions[index],
            "light_positions": self.light_positions[index],
            "elevation": self.elevation_deg[index],
            "azimuth": self.azimuth_deg[index],
            "camera_distances": self.camera_distances[index],
            "height": self.cfg.eval_height,
            "width": self.cfg.eval_width,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.cfg.eval_height, "width": self.cfg.eval_width})
        return batch

@register("random-multiview-control-depth")
class RandomMultiviewCameraDataModule(pl.LightningDataModule):
    cfg: RandomMultiviewCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomMultiviewCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset_sketch(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset_sketch(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
    
    # # Multi-process, more quickly, with small bug now
    # def train_dataloader(self) -> DataLoader:
    #     dataloader = DataLoader(
    #         self.train_dataset,
    #         # very important to disable multi-processing if you want to change self attributes at runtime!
    #         # (for example setting self.width and self.height in update_step)
    #         num_workers=16, # type: ignore
    #         batch_size=None,
    #         collate_fn=self.train_dataset.collate,
    #     )
    #     return dataloader
    
    def train_dataloader(self) -> DataLoader: 
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
    
    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
