#!/usr/bin/env python
import argparse
import logging
import math
import textwrap
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from skimage import (
    color,
    draw,
    filters,
    io,
    measure,
    morphology,
    segmentation,
    transform,
    util,
)
from skimage.filters import rank

from pylib import log

SUFFIXES = (".jpeg", ".jpg", ".png")

Circle = namedtuple("Circle", "x y radius")
Orient = namedtuple("Orient", "cos_para sin_para cos_perp sin_perp")


@dataclass
class Seed:
    dir: int = 0
    path: str | Path | None = None
    head: Any | None = None
    body: Any | None = None
    image: Any | None = None


def main():
    log.started()

    args = parse_args()

    paths = [p for p in args.image_dir.glob("*.*") if p.suffix in SUFFIXES]
    for path in sorted(paths):
        logging.info(f"Processing file {path.name}")
        segment_seeds(args, path)
    # segment_seeds(PATHS[-1])

    log.finished()


def segment_seeds(args, image_path):
    orig, gray = get_gray_scale(image_path)
    mask = get_mask(gray)
    image = enhance_contrast(gray, mask)
    edges, circles = find_circles(mask)
    big, small = circles
    edges = expand_circle(edges, small, 0, small.radius + args.bb_stop)
    edges = expand_circle(edges, big, big.radius - args.big_start, big.radius)
    edges = flood_fill(edges)
    image = threshold_otsu(image, mask)
    image = apply_mask(image, edges)
    image, labels = find_areas(image)
    seeds = join_head_body(image, image_path, labels)
    seeds = sort_seeds(seeds)
    cutout_seeds(seeds, gray, args.seed_dir)


def get_gray_scale(image_path):
    """ Get initial gray scale image."""
    orig = io.imread(image_path)
    gray = np.uint8(color.rgb2gray(orig) * 255.0)
    return orig, gray


def get_mask(gray, watermark_coords=(750, 1100)):
    """Get a mask for the image

    I don't want to perform image manipulations on the black regions of the image.
    Using this mask is important (autolevel_percentile needs it), however it introduces
    its own complications later on.
    """
    threshold = filters.threshold_otsu(gray)
    mask = gray > threshold

    mask = morphology.remove_small_objects(mask, 64)
    mask = morphology.remove_small_holes(mask, 64)

    # Remove the watermark in the lower right corner.
    # It will interfere with the image manipulations.
    rr, cc = draw.rectangle(watermark_coords, mask.shape, shape=mask.shape)
    mask[rr, cc] = 0

    return mask


def enhance_contrast(image, mask, disk=50, low=0.2, high=0.8):
    image = rank.autolevel_percentile(
        image, footprint=morphology.disk(disk), p0=low, p1=high, mask=mask
    )
    return image


def find_circles(mask, edge_threshold=0.2, small=75, large=500, count=2):
    edges = filters.sobel(mask)
    edges = edges > edge_threshold  # noqa

    hough_radii = np.arange(small, large, 4)
    hough_res = transform.hough_circle(edges, hough_radii)

    # Select the most prominent circles
    _, cx, cy, radii = transform.hough_circle_peaks(
        hough_res, hough_radii, total_num_peaks=count
    )

    circles = [Circle(y=y, x=x, radius=r) for y, x, r in zip(cy, cx, radii)]
    circles = sorted(circles, key=lambda c: -c.radius)
    return edges, circles


def expand_circle(edges, circle, start, stop, new_value=128):
    edges = edges.copy()
    for radius in range(start, stop):
        circ_y, circ_x = draw.circle_perimeter(
            circle.y, circle.x, radius, shape=edges.shape
        )
        edges[circ_y, circ_x] = new_value
    return edges


def flood_fill(edges, seed_point=(1, 1), new_value=128):
    edges = segmentation.flood_fill(edges, seed_point, new_value)
    return edges


def threshold_otsu(image, mask):
    thresh = filters.threshold_otsu(image)
    binary = image > thresh
    binary = binary & mask
    return binary


def apply_mask(image, edges, area_threshold=32, new_value=128):
    image = image.copy()
    image[edges] = new_value
    image = morphology.remove_small_holes(image, area_threshold=area_threshold)
    image = util.invert(image)
    return image


def find_areas(image):
    labels = measure.label(image)
    overlay = color.label2rgb(labels, image, bg_label=0, bg_color=None)
    return overlay, labels


def get_orientation(body):
    """Pre-calculate some orientation parameters."""
    cos_para = math.cos(body.orientation)  # parallel
    sin_para = math.sin(body.orientation)

    cos_perp = math.cos(body.orientation + math.pi / 2)  # perpendicular
    sin_perp = math.sin(body.orientation + math.pi / 2)

    return Orient(cos_para, sin_para, cos_perp, sin_perp)


def find_head(
    input_image,
    image_path,
    labels,
    body,
    body_threshold=8_000,
    head_search_height=300,
    head_axis_ratio=3.0,
):
    # Start with the centroid calculated by scikit measure.label function
    centroid_rr, centroid_cc = int(body.centroid[0]), int(body.centroid[1])
    ori = get_orientation(body)

    # Draw a line segment from the centroid along the orientation, both ways
    for dir_ in [-1, 1]:
        point_rr = int(centroid_rr + dir_ * head_search_height * ori.cos_para)
        point_cc = int(centroid_cc + dir_ * head_search_height * ori.sin_para)

        line_rr, line_cc = draw.line(centroid_rr, centroid_cc, point_rr, point_cc)

        # Find an area that intersects this line
        for head in measure.regionprops(labels):
            if head.area >= body_threshold:
                continue

            # Not a head shape
            if head.axis_major_length > head_axis_ratio * head.axis_minor_length:
                continue

            # Check intersection
            tt, ll, bb, rr = head.bbox

            line = np.zeros(input_image.shape[:-1], dtype=bool)
            line[line_rr, line_cc] = 1  # (1, 1, 1)
            line = line[tt:bb, ll:rr]

            # Does not intersect with the line, so it's not for this body
            if not (head.image & line).any():
                continue

            # Looks good return a Seed
            seed_image = np.zeros(input_image.shape[:-1], dtype=bool)

            tt, ll, bb, rr = body.bbox
            seed_image[tt:bb, ll:rr] = body.image

            tt, ll, bb, rr = head.bbox
            seed_image[tt:bb, ll:rr] = head.image

            return Seed(
                dir=dir_, path=image_path, head=head, body=body, image=seed_image
            )

    # No head found, assume it is attached or missing
    else:
        seed_image = np.zeros(input_image.shape[:-1], dtype=bool)

        tt, ll, bb, rr = body.bbox
        seed_image[tt:bb, ll:rr] = body.image

        return Seed(path=image_path, body=body, image=seed_image)


def attach_head(seed: Seed, belt_search_radius=40, head_search_height=300):
    if not seed.head:
        return

    # Start with the centroid calculated by scikit measure.label function
    centroid_rr, centroid_cc = int(seed.body.centroid[0]), int(seed.body.centroid[1])
    ori = get_orientation(seed.body)

    # Get the points of the belt of the seed, which extends from the centroid
    # to the outermost points of the seed perpendicular to the seed's orientation

    # Draw the 1st half belt
    rr = int(centroid_rr + belt_search_radius * ori.cos_perp * seed.dir)
    cc = int(centroid_cc + belt_search_radius * ori.sin_perp * seed.dir)
    belt1_rr, belt1_cc = draw.line(rr, cc, centroid_rr, centroid_cc)

    # Trim the 1st half belt
    idx = np.argwhere(seed.image[belt1_rr, belt1_cc])
    belt1_rr = int(belt1_rr[idx[0][0]])
    belt1_cc = int(belt1_cc[idx[0][0]])

    # Draw the 2nd half belt
    rr = int(centroid_rr - belt_search_radius * ori.cos_perp * seed.dir)
    cc = int(centroid_cc - belt_search_radius * ori.sin_perp * seed.dir)
    belt2_rr, belt2_cc = draw.line(rr, cc, centroid_rr, centroid_cc)

    # Trim the 2nd half belt
    idx = np.argwhere(seed.image[belt2_rr, belt2_cc])
    belt2_rr = int(belt2_rr[idx[0][0]])
    belt2_cc = int(belt2_cc[idx[0][0]])

    # Scan for the top of the head
    for i in range(head_search_height, -1, -1):
        delta_rr = centroid_rr - int(centroid_rr - i * ori.cos_para * seed.dir)
        delta_cc = centroid_cc - int(centroid_cc - i * ori.sin_para * seed.dir)

        rr1 = belt1_rr + delta_rr
        cc1 = belt1_cc + delta_cc
        rr2 = belt2_rr + delta_rr
        cc2 = belt2_cc + delta_cc

        line_rr, line_cc = draw.line(rr1, cc1, rr2, cc2)

        # Check if the line intersects the head
        tt, ll, bb, rr = seed.head.bbox
        line = np.zeros(seed.image.shape, dtype=bool)
        line[line_rr, line_cc] = 1
        head_part = line[tt:bb, ll:rr]

        if not (seed.head.image & head_part).any():
            continue

        poly_rr, poly_cc = draw.polygon(
            (rr1, rr2, belt2_rr, belt1_rr),
            (cc1, cc2, belt2_cc, belt1_cc),
        )
        seed.image[poly_rr, poly_cc] = 1

        break


def join_head_body(
    input_image,
    image_path,
    labels,
    body_threshold=5500,
    head_search_height=300,
):
    seeds = []

    input_image = input_image.copy()

    # Find head for each body, if there is one
    for body in measure.regionprops(labels):
        if body.area < body_threshold:
            continue

        seed = find_head(
            input_image, image_path, labels, body, body_threshold, head_search_height
        )
        attach_head(seed)
        seeds.append(seed)

    return seeds


def sort_seeds(seeds):
    seeds = sorted(seeds, key=lambda s: -s.body.centroid[1])
    seeds[1:] = sorted(seeds[1:], key=lambda s: s.body.centroid[0])
    return seeds


def cutout_seeds(seeds, gray, seed_dir):
    for i, seed in enumerate(seeds):
        image, labels = find_areas(seed.image)
        for props in measure.regionprops(labels):
            tt, ll, bb, rr = props.bbox
            area = gray[tt:bb, ll:rr].copy()
            area *= props.image_convex
            path = seed_dir / f"{seed.path.stem}_{i}.{seed.path.suffix}"
            io.imsave(path, area)


def parse_args():
    description = """Extract individual seeds from images."""
    arg_parser = argparse.ArgumentParser(
        description=textwrap.dedent(description), fromfile_prefix_chars="@"
    )

    arg_parser.add_argument(
        "--image-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Get images from this directory.""",
    )

    arg_parser.add_argument(
        "--seed-dir",
        type=Path,
        metavar="PATH",
        required=True,
        help="""Put segmented seed images into this directory.""",
    )

    arg_parser.add_argument(
        "--bb-stop",
        type=int,
        metavar="INT",
        default=20,
        help="""
            Expand the BB circle radius by this many pixels. (default: %(default)s)
            """,
    )

    arg_parser.add_argument(
        "--big-start",
        type=int,
        metavar="INT",
        default=32,
        help="""
            Expand the main circle radius by this many pixels. (default: %(default)s)
            """,
    )

    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    main()
