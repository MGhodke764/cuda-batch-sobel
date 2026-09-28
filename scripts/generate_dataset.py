#!/usr/bin/env python3
"""Generate a deterministic PGM image dataset using only the Python standard library."""

import argparse
import math
import os
import random


def make_image(width: int, height: int, seed: int) -> bytes:
    rng = random.Random(seed)
    data = bytearray(width * height)

    # Combine gradients, circles, rectangles, and low-amplitude noise.
    cx = width * (0.25 + 0.5 * rng.random())
    cy = height * (0.25 + 0.5 * rng.random())
    radius = min(width, height) * (0.12 + 0.18 * rng.random())

    rect_x0 = rng.randrange(width // 4)
    rect_y0 = rng.randrange(height // 4)
    rect_x1 = min(width - 1, rect_x0 + rng.randrange(width // 4, width // 2))
    rect_y1 = min(height - 1, rect_y0 + rng.randrange(height // 4, height // 2))

    for y in range(height):
        for x in range(width):
            gradient = int(80 * x / max(1, width - 1))
            wave = int(45 * (1 + math.sin(x * 0.08 + seed * 0.03)))
            distance = math.hypot(x - cx, y - cy)

            circle = 110 if distance < radius else 0
            rectangle = 80 if rect_x0 <= x <= rect_x1 and rect_y0 <= y <= rect_y1 else 0
            noise = rng.randrange(-8, 9)

            value = max(0, min(255, 20 + gradient + wave + circle + rectangle + noise))
            data[y * width + x] = value

    return bytes(data)


def write_pgm(path: str, width: int, height: int, pixels: bytes) -> None:
    with open(path, "wb") as output:
        output.write(f"P5\n{width} {height}\n255\n".encode("ascii"))
        output.write(pixels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/images")
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    for index in range(args.count):
        pixels = make_image(args.width, args.height, index + 1)
        write_pgm(
            os.path.join(args.output, f"image_{index + 1:04d}.pgm"),
            args.width,
            args.height,
            pixels,
        )

    print(
        f"Generated {args.count} PGM images "
        f"({args.width}x{args.height}) in {args.output}"
    )


if __name__ == "__main__":
    main()
