
from operator import itemgetter
import torch
import re
import collections


string_classes=str


def split_circle_central(keypoints_dict):
    # split "circle central" in  "circle central left" and "circle central right"

    # assume main camera --> TODO behind the goal camera
    if "Circle central" in keypoints_dict:
        points_circle_central_left = []
        points_circle_central_right = []

        if "Middle line" in keypoints_dict:
            p_index_ymin, _ = min(
                enumerate([p["y"] for p in keypoints_dict["Middle line"]]),
                key=itemgetter(1),
            )
            p_index_ymax, _ = max(
                enumerate([p["y"] for p in keypoints_dict["Middle line"]]),
                key=itemgetter(1),
            )
            p_ymin = keypoints_dict["Middle line"][p_index_ymin]
            p_ymax = keypoints_dict["Middle line"][p_index_ymax]
            p_xmean = (p_ymin["x"] + p_ymax["x"]) / 2

            points_circle_central = keypoints_dict["Circle central"]
            for p in points_circle_central:
                if p["x"] < p_xmean:
                    points_circle_central_left.append(p)
                else:
                    points_circle_central_right.append(p)
        else:
            # circle is partly shown on the left or right side of the image
            # mean position is shown on the left part of the image --> label right
            circle_x = [p["x"] for p in keypoints_dict["Circle central"]]
            mean_x_circle = sum(circle_x) / len(circle_x)
            if mean_x_circle < 0.5:
                points_circle_central_right = keypoints_dict["Circle central"]
            else:
                points_circle_central_left = keypoints_dict["Circle central"]

        if len(points_circle_central_left) > 0:
            keypoints_dict["Circle central left"] = points_circle_central_left
        if len(points_circle_central_right) > 0:
            keypoints_dict["Circle central right"] = points_circle_central_right
        if len(points_circle_central_left) == 0 and len(points_circle_central_right) == 0:
            raise RuntimeError
        del keypoints_dict["Circle central"]
    return keypoints_dict


def custom_list_collate(batch):
    r"""
    Function that takes in a batch of data and puts the elements within the batch
    into a tensor with an additional outer dimension - batch size. The exact output type can be
    a :class:`torch.Tensor`, a `Sequence` of :class:`torch.Tensor`, a
    Collection of :class:`torch.Tensor`, or left unchanged, depending on the input type.
    This is used as the default function for collation when
    `batch_size` or `batch_sampler` is defined in :class:`~torch.utils.data.DataLoader`.
    Here is the general input type (based on the type of the element within the batch) to output type mapping:
    * :class:`torch.Tensor` -> :class:`torch.Tensor` (with an added outer dimension batch size)
    * NumPy Arrays -> :class:`torch.Tensor`
    * `float` -> :class:`torch.Tensor`
    * `int` -> :class:`torch.Tensor`
    * `str` -> `str` (unchanged)
    * `bytes` -> `bytes` (unchanged)
    * `Mapping[K, V_i]` -> `Mapping[K, default_collate([V_1, V_2, ...])]`
    * `NamedTuple[V1_i, V2_i, ...]` -> `NamedTuple[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
    * `Sequence[V1_i, V2_i, ...]` -> `Sequence[default_collate([V1_1, V1_2, ...]), default_collate([V2_1, V2_2, ...]), ...]`
    Args:
        batch: a single batch to be collated
    Examples:
        >>> # Example with a batch of `int`s:
        >>> default_collate([0, 1, 2, 3])
        tensor([0, 1, 2, 3])
        >>> # Example with a batch of `str`s:
        >>> default_collate(['a', 'b', 'c'])
        ['a', 'b', 'c']
        >>> # Example with `Map` inside the batch:
        >>> default_collate([{'A': 0, 'B': 1}, {'A': 100, 'B': 100}])
        {'A': tensor([  0, 100]), 'B': tensor([  1, 100])}
        >>> # Example with `NamedTuple` inside the batch:
        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> default_collate([Point(0, 0), Point(1, 1)])
        Point(x=tensor([0, 1]), y=tensor([0, 1]))
        >>> # Example with `Tuple` inside the batch:
        >>> default_collate([(0, 1), (2, 3)])
        [tensor([0, 2]), tensor([1, 3])]

        >>> # modification
        >>> # Example with `List` inside the batch:
        >>> default_collate([[0, 1, 2], [2, 3, 4]])
        >>> [[0, 1, 2], [2, 3, 4]]
        >>> # original behavior
        >>> [[0, 2], [1, 3], [2, 4]]
    """

    np_str_obj_array_pattern = re.compile(r"[SaUO]")
    default_collate_err_msg_format = "default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return [torch.as_tensor(b) for b in batch]
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({key: custom_list_collate([d[key] for d in batch]) for key in elem})
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: custom_list_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(custom_list_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        # transposed = list(zip(*batch))  # It may be accessed twice, so we use a list.

        return batch

        # if isinstance(elem, tuple):
        #     return [
        #         custom_list_collate(samples) for samples in transposed
        #     ]  # Backwards compatibility.
        # else:
        #     try:
        #         return elem_type([custom_list_collate(samples) for samples in transposed])
        #     except TypeError:
        #         # The sequence type may not support `__init__(iterable)` (e.g., `range`).
        #         return [custom_list_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))