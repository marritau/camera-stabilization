# Motion Analysis Report

Variant: B, educational motion-analysis module.

Video: `data/demo_motion.mp4`
Frames processed: 140

## Lucas-Kanade

- Initial points: 200
- Final active points: 188
- Lost points: 6.0%
- Mean track length: 133.63 frames

## Farneback

- Mean dense-flow magnitude: 0.402 px
- Max mean dense-flow magnitude: 0.912 px
- Mean motion-mask area: 8.41%
- Mean connected motion components: 16.12

## Error Analysis

LK works best on corners and textured local structures. It loses tracks when an object leaves the frame,
when motion blur appears, and when local texture is weak. Farneback is better when a dense motion field
or segmentation mask is needed, but it is more sensitive to shadows, camera drift, blur, and background
texture because it estimates a vector for every pixel.
