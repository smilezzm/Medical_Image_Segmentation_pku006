import numpy as np
from scipy.ndimage import gaussian_filter

'''
This script is copied from SegVol repo, where the codes are inspired by the work of Felzenszwalb and Huttenlocher.
It aims to partition a 3D image into segments where voxels within a segment are similar in intensity.

The INPUT is a 3D numpy array.

The OUTPUT is the 3D array with the same size, values at each voxel being an integer label indicating the segment it belongs to. 
These integers are called IDs, which are consecutive integers starting from 0.

Here are key parameters:
1. scale: Controls the preference for larger or smaller segments. 
        A higher scale value leads to larger segments (less sensitive to small intensity differences). 
        It's rescaled internally (scale /= 255.0), suggesting an adaptation from an algorithm originally designed for 0-255 range images.

2. sigma: The standard deviation for a Gaussian filter applied to the image before segmentation. 
        Smoothing reduces noise and can influence the final segmentation by blurring small details. 
        A higher sigma means more smoothing.

3. min_size: A post-processing parameter. 
        Any segment smaller than min_size (in number of voxels) after the main segmentation step will be merged with a neighboring segment.
'''

def _felzenszwalb_python(image, scale=16, sigma=0.8, min_size=1000):
    def find_root(segmentation, p):
        root = p
        while (segmentation[root] < root):
            root = segmentation[root]
        return root

    def join_trees(forest, n, m):
        if (n != m):
            root = find_root(forest, n)
            root_m = find_root(forest, m)

            if (root > root_m):
                root = root_m

            set_root(forest, n, root)
            set_root(forest, m, root)

    def set_root(forest, n, root):
        while (forest[n] < n):
            j = forest[n]
            forest[n] = root
            n = j
        forest[n] = root

    if len(image.shape) != 3:
        raise ValueError("Input image should be 3D grayscale.")

    depth, height, width = image.shape

    # Rescale scale to behave like in reference implementation
    scale /= 255.0

    # Preprocess the image with Gaussian smoothing
    image = gaussian_filter(image, sigma=sigma)

    # Create the initial segmentation
    segments = np.arange(depth * width * height, dtype=np.intp).reshape(depth, height, width)
    # Initialize data structures for segment size and inner cost
    segment_size = np.ones(width * height * depth, dtype=np.intp)
    inner_cost = np.zeros(width * height * depth)

    # Compute edge weights in 8 connectivity:
    down_cost = np.abs(image[1:, :, :] - image[:-1, :, :])
    right_cost = np.abs(image[:, 1:, :] - image[:, :-1, :])
    dright_cost = np.abs(image[1:, 1:, :] - image[:-1, :-1, :])
    uright_cost = np.abs(image[1:, :-1, :] - image[:-1, 1:, :])
    front_cost = np.abs(image[:, :, 1:] - image[:, :, :-1])
    costs = np.hstack([right_cost.ravel(), down_cost.ravel(), 
                       dright_cost.ravel(), uright_cost.ravel(), 
                       front_cost.ravel()]).astype(float)
    # Compute edges between pixels
    down_edges = np.c_[segments[1:, :, :].ravel(), segments[:-1, :, :].ravel()]
    right_edges = np.c_[segments[:, 1:, :].ravel(), segments[:, :-1, :].ravel()]
    dright_edges = np.c_[segments[1:, 1:, :].ravel(), segments[:-1, :-1, :].ravel()]
    uright_edges = np.c_[segments[1:, :-1, :].ravel(), segments[:-1, 1:, :].ravel()]
    front_edges = np.c_[segments[:, :, 1:].ravel(), segments[:, :, :-1].ravel()]
    edges = np.vstack([right_edges, down_edges, 
                       dright_edges, uright_edges, 
                       front_edges])
    # Initialize the edge queue for sorting
    edge_queue = np.argsort(costs)
    edges = edges[edge_queue]
    costs = costs[edge_queue]
    segments = segments.ravel()
    # draw_cost_dist(costs)
    continuous_boundary = np.percentile(costs[costs!=0], 25)
    # Greedy iteration over edges
    for e in range(costs.size):
        seg0 = find_root(segments, edges[e, 0])
        seg1 = find_root(segments, edges[e, 1])

        if seg0 == seg1:
            continue

        # inner_cost0 = inner_cost[seg0] + scale / segment_size[seg0]
        # inner_cost1 = inner_cost[seg1] + scale / segment_size[seg1]

        if costs[e] < continuous_boundary:
        # if costs[e] < min(inner_cost0, inner_cost1):
            join_trees(segments, seg0, seg1)
            seg_new = find_root(segments, seg0)
            segment_size[seg_new] = segment_size[seg0] + segment_size[seg1]
            inner_cost[seg_new] = costs[e]

    # Post-processing to remove small segments
    for e in range(costs.size):
        seg0 = find_root(segments, edges[e, 0])
        seg1 = find_root(segments, edges[e, 1])

        if seg0 == seg1:
            continue

        if segment_size[seg0] < min_size or segment_size[seg1] < min_size:
            join_trees(segments, seg0, seg1)
            seg_new = find_root(segments, seg0)
            segment_size[seg_new] = segment_size[seg0] + segment_size[seg1]

    # Unravel the union-find tree
    flat = segments.ravel()
    old = np.zeros_like(flat)
    while (old != flat).any():
        old = flat
        flat = flat[flat]
    flat = np.unique(flat, return_inverse=True)[1]

    return flat.reshape((depth, height, width))
