import torch
import torch.nn as nn


class SAMProposalQueryEncoder(nn.Module):
    """Encode SAM proposals into BEVFormer object-query content deltas.

    Supported proposal rows:
      - 6D legacy: [xc, yc, w, h, score, class_id]
      - 13D geometry: [xc, yc, w, h, score, class_id, cam_id,
        ray_dir_x, ray_dir_y, ray_dir_z, cam_pos_x, cam_pos_y, cam_pos_z]
      - 15D mask-aware: 13D + [mask_box_fill, mask_area_frac]
    Image geometry is normalized to [0, 1]. Invalid/padded proposals should be
    masked out.
    """

    def __init__(
        self,
        embed_dims=256,
        num_classes=10,
        cls_embed_dim=32,
        hidden_dim=256,
        num_cameras=6,
        cam_embed_dim=16,
        ray_embed_dim=64,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_cameras = num_cameras
        self.class_embed = nn.Embedding(num_classes, cls_embed_dim)
        self.camera_embed = nn.Embedding(num_cameras, cam_embed_dim)
        self.ray_mlp = nn.Sequential(
            nn.Linear(6, ray_embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_embed_dim, ray_embed_dim),
            nn.ReLU(inplace=True),
        )
        self.mlp = nn.Sequential(
            nn.Linear(7 + cls_embed_dim + cam_embed_dim + ray_embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dims),
        )

    def forward(self, proposals, proposal_mask=None):
        if proposals.dim() != 3 or proposals.size(-1) not in (6, 13, 15):
            raise ValueError(f"Expected proposals [B,K,6], [B,K,13] or [B,K,15], got {tuple(proposals.shape)}")

        geom = proposals[..., :5]
        class_ids = proposals[..., 5].long().clamp(min=0, max=self.class_embed.num_embeddings - 1)
        class_feat = self.class_embed(class_ids)

        if proposals.size(-1) >= 13:
            cam_ids = proposals[..., 6].long().clamp(min=0, max=self.camera_embed.num_embeddings - 1)
            ray_inputs = proposals[..., 7:13]
        else:
            cam_ids = torch.zeros_like(class_ids)
            ray_inputs = proposals.new_zeros((*proposals.shape[:2], 6))

        if proposals.size(-1) >= 15:
            mask_quality = proposals[..., 13:15].clamp(min=0.0, max=1.0)
        else:
            mask_quality = proposals.new_zeros((*proposals.shape[:2], 2))

        camera_feat = self.camera_embed(cam_ids)
        ray_feat = self.ray_mlp(ray_inputs)
        query_delta = self.mlp(torch.cat([geom, mask_quality, class_feat, camera_feat, ray_feat], dim=-1))
        query_delta = query_delta * proposals[..., 4:5].clamp(min=0.0, max=1.0)
        if proposal_mask is not None:
            query_delta = query_delta * proposal_mask.unsqueeze(-1).to(query_delta.dtype)
        return query_delta


class SAMBEVPriorEncoder(nn.Module):
    """Build a soft BEV query bias from SAM camera rays."""

    def __init__(
        self,
        embed_dims=256,
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        bev_h=50,
        bev_w=50,
        radius=4.0,
        cam_pos_scale=100.0,
        hidden_dim=128,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.pc_range = tuple(float(v) for v in pc_range)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.radius = float(radius)
        self.cam_pos_scale = float(cam_pos_scale)
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dims),
        )

    def _grid_centers(self, device, dtype):
        x_min, y_min, _, x_max, y_max, _ = self.pc_range
        xs = torch.linspace(x_min, x_max, self.bev_w, device=device, dtype=dtype)
        ys = torch.linspace(y_min, y_max, self.bev_h, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        zz = torch.zeros_like(xx)
        return torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    def forward(self, proposals, proposal_mask=None):
        if proposals.dim() != 3 or proposals.size(-1) not in (13, 15):
            raise ValueError(f"Expected proposals [B,K,13] or [B,K,15], got {tuple(proposals.shape)}")

        bs, num_props = proposals.shape[:2]
        device, dtype = proposals.device, proposals.dtype
        grid = self._grid_centers(device, dtype)
        rays = proposals[..., 7:10]
        cam_pos = proposals[..., 10:13] * self.cam_pos_scale
        scores = proposals[..., 4].clamp(min=0.0, max=1.0)
        if proposals.size(-1) >= 15:
            quality = 0.75 + 0.25 * proposals[..., 13].clamp(min=0.0, max=1.0)
            scores = scores * quality
        if proposal_mask is not None:
            scores = scores * proposal_mask.to(dtype)

        vec = grid[None, None, :, :] - cam_pos[:, :, None, :]
        ray_norm = rays.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ray_unit = rays / ray_norm
        proj_len = (vec * ray_unit[:, :, None, :]).sum(dim=-1).clamp(min=0.0)
        closest = cam_pos[:, :, None, :] + proj_len[..., None] * ray_unit[:, :, None, :]
        dist = (grid[None, None, :, :] - closest).norm(dim=-1)
        weights = torch.exp(-0.5 * (dist / max(self.radius, 1e-6)) ** 2) * scores[:, :, None]
        prior = weights.max(dim=1).values.clamp(max=1.0)
        return self.mlp(prior.unsqueeze(-1))
