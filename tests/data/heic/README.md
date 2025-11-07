This directory contains heic test files for CrabbyAvif

## nokiatech

The files in this sub-directory are snapshotted from
https://github.com/nokiatech/heif/tree/gh-pages/content at commit
e1880923532e79090c3d12370cc02a979f94702a.

## blue_alpha.heic

HEIC image with alpha channel. Generated with ffmpeg and libheif:

ffmpeg -f lavfi -i color=c=blue@1.0:s=320x240:d=1 -frames:v 1 -pix_fmt rgba blue_alpha.png

heif-enc -o blue_alpha.heic blue_alpha.png

## blue_grid_alpha.heic

HEIC image with alpha channel in a 2x2 grid. Generated with ffmpeg and libheif:

ffmpeg -f lavfi -i color=c=blue@1.0:s=160x120:d=1 -frames:v 1 -pix_fmt rgba blue_alpha.png

cp blue_alpha.png tile-00-00.png
cp blue_alpha.png tile-00-01.png
cp blue_alpha.png tile-01-00.png
cp blue_alpha.png tile-01-01.png

heif-enc -o blue_grid_alpha.heic -T tile-00-00.png
