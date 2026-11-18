from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from spot_detection_benchmark.utils import MODES, get_device, set_piscis_dir

log = logging.getLogger(__name__)

# Model of each tool when no model is given: the tool's own pretrained model, by
# name, downloaded on first use into the tool's cache folder. In 3d mode, a tool
# without a "3d" entry uses its 2D model (see each detector's predict_3d).
DEFAULT_MODELS = {
    "spotiflow": {"2d": "general", "3d": "smfish_3d"},
    "piscis": {"2d": "20251212"},
    "ufish": {"2d": "v1.0-alldata-ufish_c32.pth"},
}

def load_spotiflow(
    model: str | Path, inference_mode: bool = True, map_location: str = "auto"
) -> Any:
    """Load a Spotiflow model from its folder or by name.

    Args:
        model: model folder, or name of a Spotiflow pretrained model (e.g.
            general, smfish_3d), downloaded once into ~/.spotiflow.
        inference_mode: False to leave the model trainable.
        map_location: device.

    Returns:
        The Spotiflow model.
    """
    from spotiflow.model import Spotiflow as SpotiflowModel

    if Path(model).is_dir():
        return SpotiflowModel.from_folder(
            model, inference_mode=inference_mode, map_location=map_location
        )
    if len(Path(model).parts) > 1:
        raise FileNotFoundError(f"spotiflow: no model folder {model}")
    return SpotiflowModel.from_pretrained(
        str(model), inference_mode=inference_mode, map_location=map_location
    )


def load_ufish_weights(ufish: Any, model: str | Path) -> None:
    """Load PyTorch weights into a U-FISH model.

    Args:
        ufish: ufish.api.UFish instance.
        model: .pth file, or name of a .pth file of U-FISH's HuggingFace
            repository (e.g. v1.0-alldata-ufish_c32.pth), downloaded once into
            ~/.ufish.

    Raises:
        ValueError: model is not a .pth model.
    """
    if not str(model).endswith(".pth"):
        raise ValueError(f"ufish: {model} is not a PyTorch .pth model")
    if Path(model).is_file():
        ufish.load_weights_from_path(str(model))
    else:
        ufish.load_weights_from_internet(weights_file=str(model))


def _sanitize_h(threshold: float, image: np.ndarray) -> int:
    h = int(round(threshold))
    if np.issubdtype(image.dtype, np.integer):
        h = min(h, np.iinfo(image.dtype).max)
    return h


def _sample(arr: np.ndarray, coords: np.ndarray) -> np.ndarray:
    if not len(coords):
        return np.array([])
    idx = tuple(
        np.clip(np.rint(coords[:, k]).astype(int), 0, arr.shape[k] - 1)
        for k in range(coords.shape[1])
    )
    return np.asarray(arr)[idx]


def _piscis_peaks(
    coords: np.ndarray, deltas: np.ndarray, radius: int = 2, tol: float = 1e-3
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, float)
    if not len(coords):
        return np.empty(coords.shape, int), np.empty(0, bool)
    yx = coords[:, -2:]
    steps = np.arange(-radius, radius + 1)
    offsets = np.stack(np.meshgrid(steps, steps, indexing="ij"), -1).reshape(-1, 2)
    candidates = np.rint(yx).astype(int)[:, None, :] + offsets[None]
    candidates = np.clip(candidates, 0, np.array(deltas.shape[-2:]) - 1)
    y, x = candidates[..., 0], candidates[..., 1]
    if coords.shape[1] == 3:
        z = np.rint(coords[:, :1]).astype(int)
        shift = deltas[z, :, y, x]
    else:
        shift = np.moveaxis(deltas[:, y, x], 0, -1)
    miss = np.abs(candidates + shift - yx[:, None, :]).max(axis=-1)
    best = miss.argmin(axis=1)
    peaks = candidates[np.arange(len(coords)), best]
    found = miss[np.arange(len(coords)), best] < tol
    if coords.shape[1] == 3:
        peaks = np.hstack((np.rint(coords[:, :1]).astype(int), peaks))
    return peaks, found


def _piscis_scores(
    coords: np.ndarray, labels: np.ndarray, deltas: np.ndarray
) -> np.ndarray:
    peaks, found = _piscis_peaks(coords, deltas)
    scores = labels[tuple(peaks.T)].astype(float)
    scores[~found] = np.nan
    if not found.all():
        log.debug(f"[piscis] {np.sum(~found)} spot(s) on the tile padding: no score")
    return scores


def _predict_stack(
    detector: Detector, stack: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray | None]:
    coords = []
    scores = []
    for i, image in enumerate(stack):
        c, s = detector.predict_2d(image, threshold)
        c = np.hstack((np.full((len(c), 1), i), c))
        coords.append(c)
        scores.append(s)

    if detector.score_name is None:
        scores = None
    if not coords:
        return np.empty((0, 3)), (None if scores is None else np.empty((0,)))
    return np.vstack(coords), (None if scores is None else np.hstack(scores))


class LoG:
    """Laplacian of Gaussian filter followed by h-maxima for spot detection.

    The threshold is the h-maxima height h, in grey levels of the image. No
    per-spot score.

    Args:
        lateral_sigma: lateral spot sigma in px.
        axial_sigma: axial spot sigma in planes (3D). Default:
            lateral_sigma * voxel_size x / voxel_size z.
        voxel_size: (z, y, x) voxel size, any unit.
    """

    score_name = None
    default_threshold = 20
    needs_model = False
    filter_modes = ()

    def __init__(
        self,
        lateral_sigma: float = 1.5,
        axial_sigma: float | None = None,
        voxel_size: Sequence[float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.lateral_sigma = lateral_sigma
        self.voxel_size = voxel_size
        self.axial_sigma = (
            lateral_sigma * voxel_size[-1] / voxel_size[0]
            if axial_sigma is None
            else axial_sigma
        )

    def predict_2d(
        self, image: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, None]:
        """Detect the spots of a 2D image.

        Args:
            image: (y, x) image.
            threshold: h-maxima height, in grey levels.

        Returns:
            (coordinates (N, 2), None).
        """
        from spot_detection_benchmark.log_detection import detect_spots_2d

        coords = detect_spots_2d(
            image, lateral_sigma=self.lateral_sigma, h=_sanitize_h(threshold, image)
        )
        return coords, None

    def predict_3d(
        self, volume: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, None]:
        """Detect the spots of a 3D volume.

        Args:
            volume: (z, y, x) volume.
            threshold: h-maxima height, in grey levels.

        Returns:
            (coordinates (N, 3), None).
        """
        from spot_detection_benchmark.log_detection import detect_spots

        coords = detect_spots(
            volume,
            axial_sigma=self.axial_sigma,
            lateral_sigma=self.lateral_sigma,
            h=_sanitize_h(threshold, volume),
        )
        return coords, None


class Piscis:
    """Piscis detector.

    The threshold is the minimum confidence of a spot. Each spot's score is
    the value of the labels map at its peak (NaN for spots whose peak lies on
    the tile padding).

    Args:
        model_path: model name, looked for in the Piscis folder (see
            set_piscis_dir); a name missing there is downloaded from Piscis's
            online models. Default: DEFAULT_MODELS.
    """

    score_name = "confidence"
    default_threshold = 0.5
    filter_modes = ("2d", "2d_stack")

    def __init__(self, model_path: str | None = None) -> None:
        set_piscis_dir()

        from piscis import Piscis as PiscisModel

        self.model = PiscisModel(
            model_name=str(model_path or DEFAULT_MODELS["piscis"]["2d"]),
            device=get_device(),
        )

    def predict_2d(
        self, image: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect the spots of a 2D image.

        Args:
            image: (y, x) image.
            threshold: minimum confidence of a spot.

        Returns:
            (coordinates (N, 2), confidences (N,)).
        """
        coords, maps = self.model.predict(
            image, threshold=threshold, intermediates=True
        )
        coords = np.asarray(coords).reshape(-1, 2)
        maps = np.asarray(maps)
        return coords, _piscis_scores(coords, maps[0], maps[1:])

    def predict_3d(
        self, volume: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect the spots of a 3D volume (Piscis stack mode).

        Args:
            volume: (z, y, x) volume.
            threshold: minimum confidence of a spot.

        Returns:
            (coordinates (N, 3), confidences (N,)).
        """
        coords, maps = self.model.predict(
            volume, stack=True, threshold=threshold, intermediates=True
        )
        coords = np.asarray(coords).reshape(-1, 3)
        maps = np.asarray(maps)
        return coords, _piscis_scores(coords, maps[:, 0], maps[:, 1:])


class UFish:
    """U-FISH detector.

    The threshold is the minimum Laplacian of the enhanced image. Each spot's
    score is that Laplacian at the spot.

    Args:
        model_path: U-FISH .pth weights, or name of U-FISH online weights (see
            load_ufish_weights). Default: DEFAULT_MODELS.
        blend_3d: 3D only, blend the enhancements along z, y and x.
    """

    score_name = "laplacian"
    default_threshold = 0.5
    filter_modes = ("2d", "2d_stack", "3d")

    def __init__(
        self, model_path: str | Path | None = None, blend_3d: bool = True
    ) -> None:
        from ufish.api import UFish as UFishModel

        self.blend_3d = blend_3d
        self.model = UFishModel()
        load_ufish_weights(self.model, model_path or DEFAULT_MODELS["ufish"]["2d"])
        on_gpu = self.model.cuda

        log.info(f"[ufish] model loaded on {'GPU' if on_gpu else 'CPU'}")
        if not on_gpu:
            log.warning(
                "[ufish] running on CPU, inference will be slow. On the cluster, use "
                "the ufish-gpu environment on a GPU node (--gres=gpu:1)."
            )

    def predict_2d(
        self, image: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect the spots of a 2D image.

        Args:
            image: (y, x) image.
            threshold: minimum Laplacian of the enhanced image.

        Returns:
            (coordinates (N, 2), Laplacians (N,)).
        """
        from skimage.filters import laplace

        spots_df, enhanced = self.model.predict(image, intensity_threshold=threshold)
        coords = spots_df[["axis-0", "axis-1"]].to_numpy()
        return coords, _sample(laplace(enhanced), coords)

    def predict_3d(
        self, volume: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect the spots of a 3D volume.

        Args:
            volume: (z, y, x) volume.
            threshold: minimum Laplacian of the enhanced volume.

        Returns:
            (coordinates (N, 3), Laplacians (N,)).
        """
        from skimage.filters import laplace

        spots_df, enhanced = self.model.predict(
            volume,
            axes="zyx",
            blend_3d=self.blend_3d,
            intensity_threshold=threshold,
        )
        coords = spots_df[["axis-0", "axis-1", "axis-2"]].to_numpy()
        return coords, _sample(laplace(enhanced), coords)


class Spotiflow:
    """Spotiflow detector, 2D or 3D model.

    The threshold is the minimum probability of a spot. Each spot's score is
    its probability. The default threshold is the model's own (thresholds.yaml).

    Args:
        model_path: Spotiflow model folder, or name of a Spotiflow pretrained
            model (see load_spotiflow). Default: DEFAULT_MODELS (2D model).
    """

    score_name = "probability"
    default_threshold = None
    filter_modes = ("2d", "2d_stack", "3d")

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.device = get_device()
        self.model = load_spotiflow(
            model_path or DEFAULT_MODELS["spotiflow"]["2d"], map_location=self.device
        )
        self.default_threshold = float(np.ravel(self.model._prob_thresh)[0])

    def predict_2d(
        self, image: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Detect the spots of a 2D image.

        Args:
            image: (y, x) image.
            threshold: minimum probability of a spot.

        Returns:
            (coordinates (N, 2), probabilities (N,)).
        """
        if self.model.config.is_3d:
            raise ValueError(
                "--mode 2d / --mode 2d_stack requires a Spotiflow model trained "
                "in 2D, but this one is 3D only. Use a 2D Spotiflow model, or "
                "--mode 3d to predict in 3D."
            )
        coords, details = self.model.predict(
            image, device=self.device, prob_thresh=threshold
        )
        return np.asarray(coords).reshape(-1, 2), getattr(details, "prob", None)

    def predict_3d(
        self, volume: np.ndarray, threshold: float
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Detect the spots of a 3D volume.

        Args:
            volume: (z, y, x) volume.
            threshold: minimum probability of a spot.

        Returns:
            (coordinates (N, 3), probabilities (N,)).
        """
        if not self.model.config.is_3d:
            raise ValueError(
                "--mode 3d requires a Spotiflow model trained in 3D, but this one is "
                "2D only. Use a 3D Spotiflow model, or --mode 2d / --mode 2d_stack "
                "to predict in 2D."
            )
        coords, details = self.model.predict(
            volume, device=self.device, prob_thresh=threshold
        )
        return np.asarray(coords).reshape(-1, 3), getattr(details, "prob", None)


DETECTORS = {
    "log": LoG,
    "spotiflow": Spotiflow,
    "piscis": Piscis,
    "ufish": UFish,
}

TOOLS = list(DETECTORS)

Detector = Union[LoG, Piscis, UFish, Spotiflow]


def default_model(tool: str, mode: str) -> str | None:
    """Name of a tool's default model (DEFAULT_MODELS).

    Args:
        tool: tool name.
        mode: "2d", "2d_stack" or "3d". In 3d mode, the 3D model if the tool
            has one, else its 2D model.

    Returns:
        The model name, or None for a tool without model (LoG).
    """
    models = DEFAULT_MODELS.get(tool)
    if models is None:
        return None
    return models.get("3d", models["2d"]) if mode == "3d" else models["2d"]


def resolve_model(
    tool: str, mode: str, model_path: str | Path | None = None
) -> str | Path | None:
    """Model a run will use.

    Args:
        tool: tool name.
        mode: "2d", "2d_stack" or "3d".
        model_path: model given by the user, returned as is.

    Returns:
        model_path, else the tool's default model. None for a tool without
        model.
    """
    if not getattr(DETECTORS[tool], "needs_model", True):
        return None
    return model_path if model_path is not None else default_model(tool, mode)


def run_name(tool: str, model_path: str | Path | None = None) -> str:
    """Name of a run, used for its results folder.

    Args:
        tool: tool name.
        model_path: model given by the user; None for the default model.

    Returns:
        "<tool>" for a tool without model, "<tool>_default" with the default
        model, else "<tool>_<model file or folder name>".
    """
    if not getattr(DETECTORS[tool], "needs_model", True):
        return tool
    if model_path is None:
        return f"{tool}_default"
    return f"{tool}_{Path(str(model_path)).stem}"


def results_dirname(tool: str, model_path: str | Path | None = None) -> str:
    """Name of the folder holding a run's prediction CSVs: "results_<run_name>".

    Args:
        tool: tool name.
        model_path: model given by the user; None for the default model.

    Returns:
        The folder name.
    """
    return f"results_{run_name(tool, model_path)}"


def make_detector(
    tool: str,
    mode: str,
    model_path: str | Path | None = None,
    **kwargs: Any,
) -> Detector:
    """Build a tool's detector, with its model loaded.

    Args:
        tool: tool name, a key of DETECTORS.
        mode: "2d", "2d_stack" or "3d", used to pick the default model.
        model_path: model to load (file, folder or name). Default: the tool's
            default model, downloaded on first use.
        **kwargs: settings of the detector class.

    Returns:
        The detector.
    """
    if tool not in DETECTORS:
        raise ValueError(f"Unknown tool: {tool}. Available tools: {TOOLS}")
    if mode not in MODES:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {MODES}")

    model_path = resolve_model(tool, mode, model_path)
    if model_path is None:
        return DETECTORS[tool](**kwargs)
    return DETECTORS[tool](model_path=model_path, **kwargs)


def detect(
    detector: Detector,
    image: np.ndarray,
    mode: str,
    threshold: float | None = None,
) -> pd.DataFrame:
    """Detect the spots of one image.

    Args:
        detector: detector built by make_detector.
        image: (y, x) image in 2d mode, (n, y, x) stack of independent planes in
            2d_stack mode, (z, y, x) volume in 3d mode.
        mode: "2d", "2d_stack" or "3d".
        threshold: detection threshold. Default: the detector's default_threshold.

    Returns:
        One row per spot: axis-0, axis-1 (and axis-2 in 2d_stack and 3d modes),
        plus a score column named after detector.score_name (NaN when no score).
    """
    if threshold is None:
        threshold = detector.default_threshold

    if mode == "2d":
        coords, scores = detector.predict_2d(image, threshold)
    elif mode == "2d_stack":
        coords, scores = _predict_stack(detector, image, threshold)
    elif mode == "3d":
        coords, scores = detector.predict_3d(image, threshold)
    else:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {MODES}")

    ndim = 2 if mode == "2d" else 3
    coords = np.asarray(coords).reshape(-1, ndim)
    spots = pd.DataFrame(coords, columns=[f"axis-{k}" for k in range(ndim)])
    if detector.score_name is not None:
        spots[detector.score_name] = (
            np.full(len(spots), np.nan)
            if scores is None
            else np.asarray(scores).reshape(-1)
        )
    return spots
