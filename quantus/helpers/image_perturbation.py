print("This file contains all the file for image perturbation based on max gradient")
import numpy as np
import itertools
from ..helpers import asserts
from typing import Callable, Dict, List, Union
from quantus.helpers.perturb_func import baseline_replacement_by_mask
from ..helpers import utils
from tqdm import tqdm
from ..helpers.perturb_func import baseline_replacement_by_indices
def threshold_perturbation(
    x_batch: np.array,
    a_batch: Union[np.array, None],
    level: int,
    sample_no: int
):
    iterator = enumerate(zip(x_batch, a_batch))
    last_results = []
    for sample, (x_batch, a_batch) in iterator:
        patch_size = 9  
        a = a_batch.copy()
        max_heatmap_val = np.max(a)
        min_heatmap_val = np.min(a)
        sub_results = []
        x_perturbed = x_batch.copy()
        mask = np.zeros(x_batch.shape, dtype=bool)
    
        for i in range(mask.shape[0]):
            mask[i] = a > (np.linspace(max_heatmap_val, min_heatmap_val, num=level)[sample_no])
       
        x_perturbed = baseline_replacement_by_mask(x_batch, mask)
        print("mask shape:",  x_perturbed.shape) 
        last_results.append(x_perturbed)
    return last_results


def region_perturbation(
        x_batch: np.array,
        a_batch: Union[np.array, None],
        patch_size: int, 
        regions_evaluation: int,
        order: None,
        sample_no: int
    ):
    # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
    a_batch = utils.expand_attribution_channel(a_batch, x_batch)
    a_axes = utils.infer_attribution_axes(a_batch, x_batch)
    iterator = enumerate(zip(x_batch, a_batch))
    last_results = []
    for sample, (x, a) in iterator:
            # Predict on input.
        patches = []
        sub_results = []
        x_perturbed = x.copy()
        # Pad input and attributions. This is needed to allow for any patch_size.
        pad_width = patch_size - 1
        x_pad = utils._pad_array(x, pad_width, mode="constant", padded_axes=a_axes)
        a_pad = utils._pad_array(a, pad_width, mode="constant", padded_axes=a_axes)

        # Create patches across whole input shape and aggregate attributions.
        att_sums = []
        axis_iterators = [
            range(pad_width, x_pad.shape[axis] - pad_width) for axis in a_axes
        ]
        for top_left_coords in itertools.product(*axis_iterators):
            # Create slice for patch.
            patch_slice = utils.create_patch_slice(
                patch_size=patch_size,
                coords=top_left_coords,
                )

                # Sum attributions for patch.
            att_sums.append(
                a_pad[utils.expand_indices(a_pad, patch_slice, a_axes)].sum()
            )  
            patches.append(patch_slice)

        if order == "random":
            # Order attributions randomly.
            order = np.arange(len(patches))
            np.random.shuffle(order)

        elif order == "morf":
            # Order attributions according to the most relevant first.
            order = np.argsort(att_sums)[::-1]

        else:
            # Order attributions according to the least relevant first.
            order = np.argsort(att_sums)

        # Create ordered list of patches.
        ordered_patches = [patches[p] for p in order]

        # Remove overlapping patches
        blocked_mask = np.zeros(x_pad.shape, dtype=bool)
        ordered_patches_no_overlap = []

        for patch_slice in ordered_patches:
            patch_mask = np.zeros(x_pad.shape, dtype=bool)
            patch_mask[utils.expand_indices(patch_mask, patch_slice, a_axes)] = True
            intersected = blocked_mask & patch_mask

            if not intersected.any():
                ordered_patches_no_overlap.append(patch_slice)
                blocked_mask = blocked_mask | patch_mask

            if len(ordered_patches_no_overlap) >= regions_evaluation:
                break

            # Increasingly perturb the input and store the decrease in function value.
        xs_perturbed =[]
        for patch_slice in ordered_patches_no_overlap:
                # Pad x_perturbed. The mode should probably depend on the used perturb_func?
                x_perturbed_pad = utils._pad_array(
                    x_perturbed, pad_width, mode="edge", padded_axes=a_axes
                )

                # Perturb.
                x_perturbed_pad = baseline_replacement_by_indices(
                    arr=x_perturbed_pad,
                    indices=patch_slice,
                    indexed_axes=a_axes,
                    perturb_baseline= "uniform"
                )

                # Remove Padding
                x_perturbed = utils._unpad_array(
                    x_perturbed_pad, pad_width, padded_axes=a_axes
                )

                asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)
                xs_perturbed.append(x_perturbed)

        last_results.append(xs_perturbed[sample_no])
    return  last_results
def irof_perturbation(
        x_batch: np.array,
        a_batch: Union[np.array, None],
        segmentation_method: None,
        sample_no:20
    ) -> List[float]:
    # Reshape input batch to channel first order.
   
    x_batch_s = x_batch.copy()

    nr_channels = x_batch_s.shape[1]
    last_results = []
    
    # Expand attributions to input dimensionality and infer input dimensions covered by the attributions.
    a_batch = utils.expand_attribution_channel(a_batch, x_batch_s)
    a_axes = utils.infer_attribution_axes(a_batch, x_batch_s)

    # Asserts.
    asserts.assert_attributions(x_batch=x_batch_s, a_batch=a_batch)

    # Use tqdm progressbar if not disabled.
  

    iterator = enumerate(zip(x_batch_s, a_batch))
    for ix, (x, a) in iterator:

        """if self.normalise:
            a = self.normalise_func(a)

        if self.abs:
            a = np.abs(a)"""

        # Predict on x. m
        # Segment image.
        segments = utils.get_superpixel_segments(
            img=np.moveaxis(x, 0, -1).astype("double"),
            segmentation_method=segmentation_method,
        )
        nr_segments = segments.max()
        asserts.assert_nr_segments(nr_segments=nr_segments)

        # Calculate average attribution of each segment.
        att_segs = np.zeros(nr_segments)
        for i, s in enumerate(range(nr_segments)):
            att_segs[i] = np.mean(a[:, segments == s])

        # Sort segments based on the mean attribution (descending order).
        s_indices = np.argsort(-att_segs)

        perturbe = []

        for i_ix, s_ix in enumerate(s_indices):

            # Perturb input by indices of attributions.
            a_ix = np.nonzero(np.repeat((segments == s_ix).flatten(), nr_channels))[
                0
            ]

            x_perturbed = baseline_replacement_by_indices(
                arr=x,
                indices=a_ix,
                indexed_axes=a_axes,
                perturb_baseline="uniform"
            )
            asserts.assert_perturbation_caused_change(x=x, x_perturbed=x_perturbed)

            # Predict on perturbed input x.
            

            # Normalise the scores to be within range [0, 1].
            perturb.append(x_perturbed)

        
        all_results.append(perturb[sample_no])

    return last_results




