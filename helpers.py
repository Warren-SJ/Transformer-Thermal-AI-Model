import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

def ensure_bgr(img):
    """Ensure the input image is in BGR format.
    args:
        img: Input image (numpy array).
    returns:
        The image in BGR format (numpy array), or None if input is None.
    """
    if img is None:
        return None
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        return cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    return img

def make_red_mask_hsv(img_bgr):
    """Return a binary mask of red regions in the input BGR image.
    args:
        img_bgr: Input image in BGR color space (numpy array).
    returns:
        A binary mask (numpy array) where red regions are white (255) and other regions are black (0).
    """
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 60], dtype=np.uint8)
    upper_red1 = np.array([12, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([168, 100, 60], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)
    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask1, mask2)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    return mask

def detect_legend_mask(img_bgr):
    """
    Return a binary mask of legend regions in the input BGR image. Legends are detected as high-saturation tall/narrow rectangles
    args:
        img_bgr: Input image in BGR color space (numpy array).
    returns:
        A binary mask (numpy array) where legend regions are white (255) and other regions are black (0).
    """
    h, w = img_bgr.shape[:2]
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    sat_mask = cv.inRange(s, 100, 255)
    vert_kernel = cv.getStructuringElement(cv.MORPH_RECT, (max(2, w // 100), max(10, h // 20)))
    sat_vert = cv.morphologyEx(sat_mask, cv.MORPH_CLOSE, vert_kernel, iterations=1)
    contours, _ = cv.findContours(sat_vert, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    legend_mask = np.zeros((h, w), dtype=np.uint8)
    for cnt in contours:
        x, y, ww, hh = cv.boundingRect(cnt)
        aspect = hh / max(1, ww)
        area = ww * hh
        is_tall_bar = (ww < 0.22 * w and aspect > 3.5)
        is_large_enough = (area > 0.01 * w * h)
        near_edge = (x > 0.65 * w) or (y > 0.65 * h)
        if (is_tall_bar and is_large_enough and near_edge) or (hh < 0.15 * h and ww > 0.4 * w and y > 0.7 * h):
            cv.rectangle(legend_mask, (x, y), (x + ww, y + hh), 255, -1)
    right_strip = int(0.12 * w)
    bottom_strip = int(0.12 * h)
    legend_mask[:, w - right_strip:] = 255
    legend_mask[h - bottom_strip:, :] = 255
    return legend_mask

def remove_small_components(mask, min_area):
    """
    Remove small connected components from the binary mask.
    args:
        mask: Input binary mask (numpy array).
        min_area: Minimum area threshold for components to keep.
    returns:
        A binary mask (numpy array) with small components removed.
    """
    num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        area = stats[i, cv.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 255
    return out

def hs_histogram(img_bgr, h_bins: int = 24, s_bins: int = 16, mask=None) -> np.ndarray:
    """Compute a normalized 2D histogram over Hue and Saturation.
    - img_bgr: input image in BGR
    - h_bins, s_bins: number of bins for H and S
    - mask: optional mask; robust to dtype/shape
    Returns a 1D float32 vector of length h_bins*s_bins summing to 1.
    """
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    H, S, _ = cv.split(hsv)
    if mask is None:
        mask_u8 = np.ones(H.shape, dtype=np.uint8) * 255
    else:
        m = mask
        if m.ndim == 3:
            m = cv.cvtColor(m, cv.COLOR_BGR2GRAY)
        if m.shape != H.shape:
            m = cv.resize(m, (H.shape[1], H.shape[0]), interpolation=cv.INTER_NEAREST)
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        vmax = int(m.max()) if m.size > 0 else 0
        if vmax <= 1:
            m = (m * 255).astype(np.uint8)
        mask_u8 = m
    hist = cv.calcHist([H, S], [0, 1], mask_u8, [h_bins, s_bins], [0, 180, 0, 256])
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    if s > 0:
        hist /= s
    return hist.flatten()

def make_warm_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Warm = red OR yellow/orange in HSV."""
    red = make_red_mask_hsv(img_bgr)
    hsv = cv.cvtColor(img_bgr, cv.COLOR_BGR2HSV)
    lower_y = np.array([15, 80, 60], dtype=np.uint8)
    upper_y = np.array([35, 255, 255], dtype=np.uint8)
    yellow = cv.inRange(hsv, lower_y, upper_y)
    warm = cv.bitwise_or(red, yellow)
    return warm

def detect_fault_regions(img_bgr: np.ndarray, baseline_bgr: np.ndarray,
                         min_area_frac: float = 0.0005) -> Tuple[np.ndarray, list]:
    """Return warm mask and list of bounding boxes for warm regions above baseline.
    min_area_frac: fraction of image area; regions below are ignored.
    args:
        img_bgr: Input BGR image (numpy array).
        baseline_bgr: Baseline BGR image (numpy array).
        min_area_frac: Minimum area fraction of image area to consider a region valid.
    returns:
        A tuple (new_warm, boxes) where new_warm is a binary mask of detected warm regions
    """
    H, W = img_bgr.shape[:2]
    base = cv.resize(baseline_bgr, (W, H), interpolation=cv.INTER_AREA)
    warm_img = make_warm_mask(img_bgr)
    warm_base = make_warm_mask(base)
    new_warm = cv.bitwise_and(warm_img, cv.bitwise_not(warm_base))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    new_warm = cv.morphologyEx(new_warm, cv.MORPH_CLOSE, kernel, iterations=1)
    new_warm = remove_small_components(new_warm, min_area=max(5, int(min_area_frac * H * W)))
    contours, _ = cv.findContours(new_warm, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        boxes.append((x, y, w, h))

    # Suppress boxes that are at least 80% contained within another box
    def _ioa(inner, outer):
        xi1 = max(inner[0], outer[0])
        yi1 = max(inner[1], outer[1])
        xi2 = min(inner[0] + inner[2], outer[0] + outer[2])
        yi2 = min(inner[1] + inner[3], outer[1] + outer[3])
        iw = max(0, xi2 - xi1)
        ih = max(0, yi2 - yi1)
        inter = iw * ih
        inner_area = max(1, inner[2] * inner[3])
        return inter / inner_area

    keep = [True] * len(boxes)
    for i in range(len(boxes)):
        if not keep[i]:
            continue
        for j in range(len(boxes)):
            if i == j:
                continue
            ioa = _ioa(boxes[i], boxes[j])
            if ioa >= 0.8:
                # i is 80% within j -> drop i (prefer keeping larger by checking area)
                area_i = boxes[i][2] * boxes[i][3]
                area_j = boxes[j][2] * boxes[j][3]
                if area_i <= area_j:
                    keep[i] = False
                    break
                else:
                    keep[j] = False
    boxes = [b for k, b in zip(keep, boxes) if k]
    return new_warm, boxes

def draw_boxes(img_bgr: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """Draw red rectangles for provided boxes on a copy of the image."""
    vis = img_bgr.copy()
    for (x, y, w, h) in boxes:
        cv.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return vis

def compute_baseline_for_transformer(t_dir: Path) -> np.ndarray:
    """Compute a baseline image for a transformer (per pixel median of normal images).
    Returns BGR uint8 image or raises if none found.
    args:
        t_dir: Path to transformer directory containing 'normal' subdirectory.
    returns:        Baseline BGR image as np.ndarray of dtype uint8.
    raises:        FileNotFoundError: if no normal images found in t_dir/normal.
    """
    normals = sorted((t_dir / 'normal').glob('*.*'))
    imgs = []
    for p in normals:
        img = cv.imread(str(p), cv.IMREAD_UNCHANGED)
        if img is None:
            continue
        imgs.append(ensure_bgr(img))
    if not imgs:
        raise FileNotFoundError(f'No normal images found in {t_dir}/normal')
    h_min = min(im.shape[0] for im in imgs)
    w_min = min(im.shape[1] for im in imgs)
    stack = [cv.resize(im, (w_min, h_min), interpolation=cv.INTER_AREA) for im in imgs]
    median = np.median(np.stack(stack, axis=0), axis=0).astype(np.uint8)
    return median

def view_misclassified_images(results, expected_label=None, expected_labels=None,
                                max_images=20, cols=5):
    """Display ONLY misclassified images.

    Parameters
    ----------
    results : dict
        Output dict from a classify_folder* function. Must contain 'list' (pred labels)
        and 'paths' (image paths, same order).
    expected_label : str, optional
        If all images in this results set share the same ground-truth label
        (e.g. 'Faulty' when results came from the faulty folder). Ignored if
        expected_labels is provided.
    expected_labels : sequence|dict, optional
        Either (a) a list/tuple with per-image ground-truth labels of the same
        length as results['list'] OR (b) a dict mapping path -> ground-truth label.
        If provided, overrides expected_label.
    max_images : int
        Maximum number of misclassified samples to visualize.
    cols : int
        Number of subplot columns.

    Behavior
    --------
    Computes ground-truth labels, finds indices where prediction != ground truth,
    and plots only those images. If there are none, prints a message.
    """
    preds = results.get('list', [])
    paths = results.get('paths', [])
    if not preds or not paths:
        print("Results lacks predictions or paths; nothing to display.")
        return
    if len(preds) != len(paths):
        print("Mismatch between number of predictions and paths; aborting.")
        return

    if expected_labels is not None:
        if isinstance(expected_labels, dict):
            gts = [expected_labels.get(p, expected_label) for p in paths]
        else:
            gts = list(expected_labels)
            if len(gts) != len(preds):
                raise ValueError("expected_labels length doesn't match predictions")
    else:
        if expected_label is None:
            raise ValueError("Provide expected_label or expected_labels.")
        gts = [expected_label] * len(preds)

    mis_idx = [i for i, (pred, gt) in enumerate(zip(preds, gts)) if pred != gt]
    total = len(preds)
    mis_total = len(mis_idx)
    print(f"Total images: {total} | Misclassified: {mis_total}")
    if mis_total == 0:
        print("No misclassifications.")
        return

    N = min(max_images, mis_total)
    rows = (N + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for j in range(rows * cols):
        ax = axes[j]
        if j < N:
            idx = mis_idx[j]
            p = paths[idx]
            img = cv.imread(p, cv.IMREAD_UNCHANGED)
            if img is None:
                ax.text(0.5, 0.5, 'Read error', ha='center', va='center')
            else:
                bgr = ensure_bgr(img)
                rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
                ax.imshow(rgb)
            pred = preds[idx]
            gt = gts[idx]
            ax.set_title(f"GT:{gt} | Pred:{pred}", fontsize=8)
            if 'shorten_path' in globals():
                ax.set_xlabel(shorten_path(p), fontsize=7)
            ax.set_xticks([]); ax.set_yticks([])
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def shorten_path(p, max_len=70):
    return p if len(p) <= max_len else ("…" + p[-(max_len-1):])

def classify_folder(folder_glob_pattern, classifier, allow_labels=("Faulty","Normal"), silent_errors=True, sort=True, **classifier_kwargs):
    """Generic folder classification helper.

    Parameters
    ----------
    folder_glob_pattern : str
        Glob pattern relative to current working directory, e.g. 'separated_data/faulty/*.*'.
    classifier : callable
        Function taking (image_path, **classifier_kwargs) returning a label string.
    allow_labels : iterable
        Labels to count explicitly; others ignored (but still stored in list if appear).
    silent_errors : bool
        If True, swallow exceptions per file (increments 'errors'); if False, raise.
    sort : bool
        Whether to sort matched paths for deterministic ordering.
    **classifier_kwargs : dict
        Extra keyword args forwarded to classifier.

    Returns
    -------
    dict with keys:
        Faulty, Normal, Total, list, paths, errors
    """
    paths_iter = Path().glob(folder_glob_pattern)
    paths = [str(p) for p in paths_iter]
    if sort:
        paths.sort()
    results = {"Faulty":0, "Normal":0, "Total":0, "list":[], "paths":[], "errors":0}
    for p in paths:
        try:
            label = classifier(p, **classifier_kwargs)
        except Exception:
            results["errors"] += 1
            if not silent_errors:
                raise
            continue
        if label in allow_labels:
            results[label] += 1
        results["Total"] += 1
        results["list"].append(label)
        results["paths"].append(p)
    return results

def classify_folder_dual(folder_glob_pattern_faulty, folder_glob_pattern_normal, classifier, **kwargs):
    """Convenience: classify faulty + normal folders separately with same classifier.

    Returns tuple: (results_faulty, results_normal)
    """
    res_faulty = classify_folder(folder_glob_pattern_faulty, classifier, **kwargs)
    res_normal = classify_folder(folder_glob_pattern_normal, classifier, **kwargs)
    return res_faulty, res_normal

def view_misclassified_images_cnn(misclassified_paths, max_images=20, cols=5):
    """Display ONLY misclassified images.
    Parameters
    ----------
    misclassified_paths : list
        List of file paths to misclassified images.
    max_images : int
        Maximum number of misclassified samples to visualize.
    cols : int
        Number of subplot columns.
    Behavior
    --------
    Plots only the misclassified images. If there are none, prints a message.
    """
    if not misclassified_paths:
        print("No misclassified images to display.")
        return

    misclassified_paths = misclassified_paths[:max_images]

    rows = (len(misclassified_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.2))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i, ax in enumerate(axes):
        if i < len(misclassified_paths):
            img = cv.imread(misclassified_paths[i])
            if img is not None:
                ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
            ax.set_title(f"Misclassified {i+1}")
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def filter_boxes_by_containment(boxes, overlap_thresh=0.8):
    """
    Remove boxes that are largely inside another box.
    Boxes are tuples (x, y, w, h).
    overlap_thresh = fraction of area that must be inside another box to remove.
    """
    if not boxes:
        return boxes

    keep = []
    for i, box_a in enumerate(boxes):
        xa, ya, wa, ha = box_a
        area_a = wa * ha
        xa2, ya2 = xa + wa, ya + ha
        contained = False
        for j, box_b in enumerate(boxes):
            if i == j:
                continue
            xb, yb, wb, hb = box_b
            xb2, yb2 = xb + wb, yb + hb

            # intersection
            inter_x1 = max(xa, xb)
            inter_y1 = max(ya, yb)
            inter_x2 = min(xa2, xb2)
            inter_y2 = min(ya2, yb2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h

            if inter_area / area_a >= overlap_thresh:
                contained = True
                break
        if not contained:
            keep.append(box_a)
    return keep
