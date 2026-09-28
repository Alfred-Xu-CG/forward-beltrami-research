"""Isolated experimental reuse of DHR's SuperPoint/SuperGlue network.

This duplicates only DHR 1.0.1's matcher call path; it does not edit the
installed package.  The cache key includes every network setting, both weight
paths and the target device.  Transform selection stays in DHR's own helpers.
"""

from __future__ import annotations

from time import perf_counter

import deeperhistreg  # noqa: F401 - installs DHR's legacy import paths
import superpoint_superglue as dhr_matcher


class CachedDHRMatcher:
    def __init__(self) -> None:
        self._models = {}
        self.calls = 0
        self.model_builds = 0
        self.model_build_seconds = 0.0
        self.inference_seconds = 0.0

    def perform_registration(self, source, target, params):
        module = dhr_matcher
        defaults = {
            "superpoint_weights_path": module.p.superpoint_model_path,
            "superglue_weights_path": module.p.superglue_model_path,
            "nms_radius": 4,
            "keypoint_threshold": 0.005,
            "max_keypoints": 3000,
            "sinkhorn_iterations": 30,
            "match_threshold": 0.3,
            "show": False,
            "echo": True,
            "transform_type": "affine",
            "device": "cuda:0",
        }
        args = {**defaults, **params}
        key = (
            str(args["superpoint_weights_path"]),
            str(args["superglue_weights_path"]),
            args["nms_radius"], args["keypoint_threshold"], args["max_keypoints"],
            args["sinkhorn_iterations"], args["match_threshold"], str(args["device"]),
        )
        config = {
            "superpoint": {
                "nms_radius": args["nms_radius"],
                "keypoint_threshold": args["keypoint_threshold"],
                "max_keypoints": args["max_keypoints"],
            },
            "superglue": {
                "weights": "outdoor",  # DHR 1.0.1's hard-coded configuration
                "sinkhorn_iterations": args["sinkhorn_iterations"],
                "match_threshold": args["match_threshold"],
            },
        }
        model = self._models.get(key)
        if model is None:
            start = perf_counter()
            model = module.sg.Matching(config).eval().to(args["device"])
            model.superpoint.load_state_dict(module.tc.load(args["superpoint_weights_path"]))
            model.superglue.load_state_dict(module.tc.load(args["superglue_weights_path"]))
            self._models[key] = model
            self.model_builds += 1
            self.model_build_seconds += perf_counter() - start

        source_tensor = module.tc.from_numpy(source).unsqueeze(0).unsqueeze(0).to(args["device"])
        target_tensor = module.tc.from_numpy(target).unsqueeze(0).unsqueeze(0).to(args["device"])
        start = perf_counter()
        prediction = model({"image0": source_tensor, "image1": target_tensor})
        prediction = {k: v[0].detach().cpu().numpy() for k, v in prediction.items()}
        self.inference_seconds += perf_counter() - start
        self.calls += 1

        points0, points1 = prediction["keypoints0"], prediction["keypoints1"]
        if args["echo"]:
            print(f"Number of source keypoints: {len(points0)}")
            print(f"Number of target keypoints: {len(points1)}")
        matches = prediction["matches0"]
        valid = matches > -1
        matched0 = points0[valid]
        matched1 = points1[matches[valid]]

        if args["show"]:
            plot = module.plt
            plot.figure()
            plot.subplot(1, 2, 1)
            plot.imshow(source_tensor.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
            plot.plot(points0[:, 0], points0[:, 1], "r*")
            plot.subplot(1, 2, 2)
            plot.imshow(target_tensor.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
            plot.plot(points1[:, 0], points1[:, 1], "r*")
            plot.figure()
            plot.subplot(1, 2, 1)
            plot.imshow(source_tensor.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
            plot.plot(matched0[:, 0], matched0[:, 1], "r*")
            plot.subplot(1, 2, 2)
            plot.imshow(target_tensor.detach().cpu().numpy()[0, 0, :, :], cmap="gray")
            plot.plot(matched1[:, 0], matched1[:, 1], "r*")
            plot.figure()
            for index in range(len(matched0)):
                plot.plot([matched0[index, 0], matched1[index, 0]],
                          [matched0[index, 1], matched1[index, 1]], "*-")
            plot.show()

        homogeneous0 = module.u.points_to_homogeneous_representation(matched0)
        homogeneous1 = module.u.points_to_homogeneous_representation(matched1)
        if args["echo"]:
            print(f"Number of matches: {len(homogeneous0)}")
        if args["transform_type"] == "affine":
            transform = module.u.calculate_affine_transform(homogeneous1, homogeneous0)
        elif args["transform_type"] == "rigid":
            transform = module.u.calculate_rigid_transform(matched1, matched0)
        else:
            raise ValueError("Unsupported transform type (rigid or affine only).")
        return transform, len(homogeneous0), matched1


def install_cached_matcher() -> CachedDHRMatcher:
    """Patch only this Python process's DHR matcher entry point."""
    cache = CachedDHRMatcher()
    dhr_matcher.perform_registration = cache.perform_registration
    return cache
