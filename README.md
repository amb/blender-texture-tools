# blender-texture-tools
Various utilities for handling images inside Blender. Currently has two types of seamless operations.

The patching algorithm tries to fill the seam locations with square patches. It's like building a puzzle that has missing pieces. You copy the existing pieces and try to fill the empty slots with them. Window is the puzzle piece size. Lines is how many puzzle piece lines horizontally and vertically there are. Overlap is how much the pieces overlap. Samples is how long the algorithm searches for matches.

The fast algorithm has one parameter: blending. Zero blending means the images are just tiled diagonally. With blending, the images are blended at the edges.
