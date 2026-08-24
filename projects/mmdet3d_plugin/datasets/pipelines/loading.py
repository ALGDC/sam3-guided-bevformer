import json
import numpy as np
from mmdet.datasets.builder import PIPELINES


CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


CLASS_TO_ID = {
    "car": 0,
    "truck": 1,
    "construction_vehicle": 2,
    "bus": 3,
    "trailer": 4,
    "barrier": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "pedestrian": 8,
    "traffic_cone": 9,
}


@PIPELINES.register_module()
class LoadSAM3Proposals(object):
    """Load SAM3 2D proposals into img_metas for query-level experiments.

    The proposal format consumed by BEVFormerHead is:
    [xc, yc, w, h, score, class_id, cam_id, ray_dir_xyz, cam_pos_xyz].
    The first four values are normalized to the original camera image.
    Multi-camera proposals are concatenated and sorted by score.
    """

    def __init__(
        self,
        sam3_results,
        max_proposals=50,
        min_score=0.0,
        camera_order=CAMERAS,
        cam_pos_scale=100.0,
        with_geometry=True,
        proposal_dim=13,
    ):
        self.sam3_results = sam3_results
        self.max_proposals = max_proposals
        self.min_score = min_score
        self.camera_order = tuple(camera_order)
        self.cam_pos_scale = float(cam_pos_scale)
        self.with_geometry = with_geometry
        if proposal_dim not in (6, 13, 15):
            raise ValueError(f"proposal_dim must be 6, 13 or 15, got {proposal_dim}")
        self.proposal_dim = proposal_dim
        with open(sam3_results, "r") as f:
            payload = json.load(f)
        self.detections = payload["detections"]

    def _image_size_from_meta(self, results, cam_idx):
        for key in ("ori_shape", "img_shape"):
            shape = results.get(key, None)
            if shape is None:
                continue
            if len(shape) >= 2 and isinstance(shape[0], (int, np.integer)):
                h, w = shape[:2]
                return float(w), float(h)
            if len(shape) > cam_idx:
                h, w = shape[cam_idx][:2]
                return float(w), float(h)
        return 1.0, 1.0

    def _camera_geometry(self, results, cam_idx, x_px, y_px):
        if not self.with_geometry:
            return [0.0] * 6

        intrinsics = results.get("cam_intrinsic", results.get("cam2img", None))
        lidar2cams = results.get("lidar2cam", None)
        if intrinsics is None or lidar2cams is None:
            return [0.0] * 6
        if cam_idx >= len(intrinsics) or cam_idx >= len(lidar2cams):
            return [0.0] * 6

        try:
            intrinsic = np.asarray(intrinsics[cam_idx], dtype=np.float32)[:3, :3]
            lidar2cam = np.asarray(lidar2cams[cam_idx], dtype=np.float32)
            cam2lidar = np.linalg.inv(lidar2cam)
            ray_cam = np.linalg.inv(intrinsic).dot(np.array([x_px, y_px, 1.0], dtype=np.float32))
            ray_lidar = cam2lidar[:3, :3].dot(ray_cam)
            ray_norm = np.linalg.norm(ray_lidar)
            if ray_norm > 1e-6:
                ray_lidar = ray_lidar / ray_norm
            cam_pos = cam2lidar[:3, 3] / max(self.cam_pos_scale, 1e-6)
            return [float(v) for v in np.concatenate([ray_lidar, cam_pos], axis=0)]
        except (np.linalg.LinAlgError, ValueError):
            return [0.0] * 6

    def _mask_quality(self, det, bw, bh):
        mask_rle = det.get("mask_rle")
        if not mask_rle:
            return [0.0, 0.0]
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            return [0.0, 0.0]

        try:
            rle = dict(mask_rle)
            counts = rle.get("counts")
            if isinstance(counts, str):
                rle["counts"] = counts.encode("ascii")
            mask_area = float(mask_utils.area(rle))
            image_h, image_w = rle.get("size", [0, 0])
            image_area = max(float(image_h) * float(image_w), 1.0)
            box_area = max(float(bw) * float(bh), 1.0)
            box_fill = min(mask_area / box_area, 1.0)
            area_frac = min(mask_area / image_area, 1.0)
            return [box_fill, area_frac]
        except Exception:
            return [0.0, 0.0]

    def __call__(self, results):
        sample_token = results.get("sample_idx")
        sample_dets = self.detections.get(sample_token, {})
        rows = []
        for cam_idx, cam in enumerate(self.camera_order):
            width, height = self._image_size_from_meta(results, cam_idx)
            for det in sample_dets.get(cam, []):
                score = float(det.get("score", 1.0))
                if score < self.min_score:
                    continue
                label = det.get("label")
                if label not in CLASS_TO_ID:
                    continue
                x1, y1, x2, y2 = [float(v) for v in det["bbox"]]
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                if bw <= 1.0 or bh <= 1.0:
                    continue
                xc = (x1 + x2) * 0.5
                yc = (y1 + y2) * 0.5
                row = [
                    xc / max(width, 1.0),
                    yc / max(height, 1.0),
                    bw / max(width, 1.0),
                    bh / max(height, 1.0),
                    score,
                    float(CLASS_TO_ID[label]),
                ]
                if self.proposal_dim >= 13:
                    row.extend([float(cam_idx), *self._camera_geometry(results, cam_idx, xc, yc)])
                if self.proposal_dim >= 15:
                    row.extend(self._mask_quality(det, bw, bh))
                rows.append(row)

        rows = sorted(rows, key=lambda x: x[4], reverse=True)
        if self.max_proposals is not None:
            rows = rows[: self.max_proposals]

        proposals = np.asarray(rows, dtype=np.float32).reshape(-1, self.proposal_dim)
        results["sam_proposals"] = proposals
        results["sam_proposal_mask"] = np.ones((proposals.shape[0],), dtype=np.bool_)
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(sam3_results={self.sam3_results}, "
            f"max_proposals={self.max_proposals}, min_score={self.min_score}, "
            f"with_geometry={self.with_geometry}, proposal_dim={self.proposal_dim})"
        )


@PIPELINES.register_module()
class LoadSAM3MaskPriors(object):
    """Build low-resolution per-camera foreground priors from SAM3 masks."""

    def __init__(
        self,
        sam3_results,
        min_score=0.30,
        camera_order=CAMERAS,
        prior_size=(32, 56),
        score_power=1.0,
        max_area_frac=0.60,
    ):
        self.sam3_results = sam3_results
        self.min_score = float(min_score)
        self.camera_order = tuple(camera_order)
        self.prior_h = int(prior_size[0])
        self.prior_w = int(prior_size[1])
        self.score_power = float(score_power)
        self.max_area_frac = float(max_area_frac)
        with open(sam3_results, "r") as f:
            payload = json.load(f)
        self.detections = payload["detections"]

    def _decode_mask(self, mask_rle):
        if not mask_rle:
            return None
        try:
            from pycocotools import mask as mask_utils
        except ImportError:
            return None
        try:
            rle = dict(mask_rle)
            counts = rle.get("counts")
            if isinstance(counts, str):
                rle["counts"] = counts.encode("ascii")
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                mask = mask[..., 0]
            return mask.astype(np.float32)
        except Exception:
            return None

    def _resize_mask(self, mask):
        try:
            import cv2
            return cv2.resize(mask, (self.prior_w, self.prior_h), interpolation=cv2.INTER_AREA)
        except ImportError:
            y_idx = np.linspace(0, mask.shape[0] - 1, self.prior_h).astype(np.int64)
            x_idx = np.linspace(0, mask.shape[1] - 1, self.prior_w).astype(np.int64)
            return mask[np.ix_(y_idx, x_idx)]

    def __call__(self, results):
        sample_token = results.get("sample_idx")
        sample_dets = self.detections.get(sample_token, {})
        priors = np.zeros((len(self.camera_order), self.prior_h, self.prior_w), dtype=np.float32)

        for cam_idx, cam in enumerate(self.camera_order):
            for det in sample_dets.get(cam, []):
                score = float(det.get("score", 0.0))
                if score < self.min_score:
                    continue
                mask = self._decode_mask(det.get("mask_rle"))
                if mask is None:
                    continue
                mask_mean = float(mask.mean())
                if mask_mean <= 0.0 or mask_mean > self.max_area_frac:
                    continue
                weight = score ** self.score_power
                resized = self._resize_mask(mask)
                priors[cam_idx] = np.maximum(priors[cam_idx], resized * weight)

        priors = np.clip(priors, 0.0, 1.0)
        results["sam_mask_priors"] = priors
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(sam3_results={self.sam3_results}, "
            f"min_score={self.min_score}, prior_size=({self.prior_h}, {self.prior_w}), "
            f"score_power={self.score_power}, max_area_frac={self.max_area_frac})"
        )
