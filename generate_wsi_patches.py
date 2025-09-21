"""Utility script for segmenting a whole-slide image (WSI) and extracting patches.

This script loads a single WSI, segments the tissue region to obtain a tissue mask,
optionally writes a visualization of the mask, and extracts patches that are stored
in an HDF5 file.  The functionality is powered by the existing CLAM
``WholeSlideImage`` helper class so that the behaviour is consistent with the rest
of the code base.

Example usage::

    python generate_wsi_patches.py \
        --wsi-path /path/to/slide.svs \
        --patch-output-dir /path/to/output/patches \
        --mask-output-dir /path/to/output/masks

"""
from __future__ import annotations

import argparse
import os
import time
from typing import List

from wsi_core.WholeSlideImage import WholeSlideImage


def _parse_id_list(raw: str | None) -> List[int]:
    """Parse a comma separated list of integers."""
    if raw is None:
        return []

    raw = raw.strip()
    if not raw or raw.lower() == "none":
        return []

    return [int(item) for item in raw.split(",") if item]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment a whole-slide image, save the resulting tissue mask, and "
            "extract patches into an HDF5 bag."
        )
    )
    parser.add_argument(
        "--wsi-path",
        required=True,
        help="Path to the whole-slide image to process.",
    )
    parser.add_argument(
        "--patch-output-dir",
        required=True,
        help="Directory where the patch HDF5 file will be written.",
    )
    parser.add_argument(
        "--mask-output-dir",
        required=True,
        help="Directory where the segmentation mask assets will be stored.",
    )
    parser.add_argument(
        "--patch-level",
        type=int,
        default=0,
        help="OpenSlide level from which to sample the patches (default: 0).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Spatial size of the extracted patches (default: 256).",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=256,
        help="Step size between consecutive patches (default: 256).",
    )
    parser.add_argument(
        "--custom-downsample",
        type=int,
        default=1,
        help=(
            "Optional downsample factor applied to extracted patches before saving. "
            "Set to 2 to downsample by a factor of 2 (default: 1)."
        ),
    )
    parser.add_argument(
        "--save-coord",
        action="store_true",
        help="If set, store patch coordinates in the HDF5 output.",
    )
    parser.add_argument(
        "--white-thresh",
        type=int,
        default=15,
        help="Saturation threshold for filtering white patches (default: 15).",
    )
    parser.add_argument(
        "--black-thresh",
        type=int,
        default=50,
        help="Intensity threshold for filtering black patches (default: 50).",
    )
    parser.add_argument(
        "--disable-patch-filtering",
        action="store_true",
        help="Disable white/black patch filtering when extracting patches.",
    )
    parser.add_argument(
        "--seg-level",
        type=int,
        default=-1,
        help="OpenSlide level used for tissue segmentation (-1 selects automatically).",
    )
    parser.add_argument(
        "--seg-sthresh",
        type=int,
        default=8,
        help="S channel threshold when not using Otsu segmentation (default: 8).",
    )
    parser.add_argument(
        "--seg-sthresh-up",
        type=int,
        default=255,
        help="Upper bound for the saturation threshold (default: 255).",
    )
    parser.add_argument(
        "--seg-mthresh",
        type=int,
        default=7,
        help="Median filter kernel size applied prior to thresholding (default: 7).",
    )
    parser.add_argument(
        "--seg-close",
        type=int,
        default=4,
        help="Kernel size used for morphological closing (default: 4).",
    )
    parser.add_argument(
        "--use-otsu",
        action="store_true",
        help="Enable Otsu thresholding when segmenting tissue.",
    )
    parser.add_argument(
        "--filter-area",
        type=float,
        default=100.0,
        help="Minimum tissue area (at reference scale) to keep during segmentation.",
    )
    parser.add_argument(
        "--filter-hole-area",
        type=float,
        default=16.0,
        help="Minimum hole area (at reference scale) to keep during segmentation.",
    )
    parser.add_argument(
        "--filter-max-n-holes",
        type=int,
        default=8,
        help="Maximum number of holes kept per contour (default: 8).",
    )
    parser.add_argument(
        "--ref-patch-size",
        type=int,
        default=512,
        help="Reference patch size used to scale the area thresholds (default: 512).",
    )
    parser.add_argument(
        "--mask-vis-level",
        type=int,
        default=-1,
        help="OpenSlide level used when saving the mask visualization (-1 selects automatically).",
    )
    parser.add_argument(
        "--mask-downsample",
        type=int,
        default=1,
        help="Downsample factor applied to the saved mask visualization (default: 1).",
    )
    parser.add_argument(
        "--no-mask-visualization",
        action="store_true",
        help="Skip writing the PNG overlay of the tissue mask.",
    )
    parser.add_argument(
        "--keep-ids",
        type=str,
        default=None,
        help="Optional comma separated contour ids to keep during segmentation.",
    )
    parser.add_argument(
        "--exclude-ids",
        type=str,
        default=None,
        help="Optional comma separated contour ids to exclude during segmentation.",
    )
    parser.add_argument(
        "--contour-fn",
        type=str,
        default="four_pt",
        choices=["four_pt", "four_pt_hard", "center", "basic"],
        help="Contour inclusion rule used when extracting patches (default: four_pt).",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=250,
        help="Line thickness (in visualization pixels) for the mask overlay.",
    )
    parser.add_argument(
        "--number-contours",
        action="store_true",
        help="Annotate contour indices on the saved mask visualization.",
    )
    return parser


def _auto_select_level(level: int, wsi: WholeSlideImage) -> int:
    """Choose an appropriate OpenSlide level when the user passes ``-1``."""
    if level >= 0:
        return level

    if len(wsi.level_dim) == 1:
        return 0

    openslide_obj = wsi.getOpenSlide()
    return openslide_obj.get_best_level_for_downsample(64)


def segment_wsi(args: argparse.Namespace, wsi: WholeSlideImage) -> str:
    """Run tissue segmentation and persist the resulting mask assets."""
    seg_level = _auto_select_level(args.seg_level, wsi)

    filter_params = {
        "a_t": args.filter_area,
        "a_h": args.filter_hole_area,
        "max_n_holes": args.filter_max_n_holes,
    }

    keep_ids = _parse_id_list(args.keep_ids)
    exclude_ids = _parse_id_list(args.exclude_ids)

    start = time.time()
    wsi.segmentTissue(
        seg_level=seg_level,
        sthresh=args.seg_sthresh,
        sthresh_up=args.seg_sthresh_up,
        mthresh=args.seg_mthresh,
        close=args.seg_close,
        use_otsu=args.use_otsu,
        filter_params=filter_params,
        ref_patch_size=args.ref_patch_size,
        keep_ids=keep_ids,
        exclude_ids=exclude_ids,
    )
    elapsed = time.time() - start

    os.makedirs(args.mask_output_dir, exist_ok=True)
    mask_asset_path = os.path.join(args.mask_output_dir, f"{wsi.name}_mask.pkl")
    wsi.saveSegmentation(mask_asset_path)
    print(f"Saved segmentation mask contours to {mask_asset_path}")
    print(f"Segmentation completed in {elapsed:.2f} seconds using level {seg_level}.")

    if not args.no_mask_visualization:
        vis_level = _auto_select_level(args.mask_vis_level, wsi)
        overlay = wsi.visWSI(
            vis_level=vis_level,
            line_thickness=args.line_thickness,
            custom_downsample=args.mask_downsample,
            number_contours=args.number_contours,
            seg_display=True,
            annot_display=False,
        )
        overlay_path = os.path.join(args.mask_output_dir, f"{wsi.name}_mask.png")
        overlay.save(overlay_path)
        print(f"Saved mask visualization to {overlay_path}")

    return mask_asset_path


def extract_patches(args: argparse.Namespace, wsi: WholeSlideImage) -> str:
    """Extract patches inside the segmented tissue mask and write them to HDF5."""
    os.makedirs(args.patch_output_dir, exist_ok=True)

    patch_file = wsi.createPatches_bag_hdf5(
        save_path=args.patch_output_dir,
        patch_level=args.patch_level,
        patch_size=args.patch_size,
        step_size=args.step_size,
        save_coord=args.save_coord,
        custom_downsample=args.custom_downsample,
        white_black=not args.disable_patch_filtering,
        white_thresh=args.white_thresh,
        black_thresh=args.black_thresh,
        contour_fn=args.contour_fn,
    )

    print(f"Extracted patches saved to {patch_file}")
    return patch_file


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if not os.path.isfile(args.wsi_path):
        raise FileNotFoundError(f"WSI path does not exist: {args.wsi_path}")

    wsi = WholeSlideImage(args.wsi_path)
    print(f"Processing slide: {wsi.name}")

    segment_wsi(args, wsi)
    extract_patches(args, wsi)


if __name__ == "__main__":
    main()
