import numpy as np
import torch
from random import shuffle


def points(p: torch.Tensor):
    directions = np.array([
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1]
    ])
    new_points = p + directions
    return new_points

def valid_points(points, width, height):
    for x, y in points:
        if x < 0 or x >= width:
            continue
        if y < 0 or y >= height:
            continue
        yield x, y

def get_points(p, width, height):
    new_points = points(p)
    yield from valid_points(new_points, width, height)


def flood_fill_loss(recon, target, threshold = 0.1, return_shapes=False):
    batch, _, width, height = target.shape

    loss = 0

    batched_shapes = segment(target, threshold=threshold)
    for b in range(batch):

        print(f'batch {b} has {len(batched_shapes[b])} shapes')
        for val, coords in batched_shapes[b]:

            # print(f'batch {b}, group has {len(coords)} members')
            recon_values = []
            target_values = []

            for x, y in coords:
                recon_values.append(recon[b, :, x, y].view(1))
                target_values.append(target[b, :, x, y].view(1))

            recon_values = torch.cat(recon_values)
            target_values = torch.cat(target_values)

            # each group/shape count equally, regardless of number of points
            loss = loss + torch.abs(recon_values - target_values).mean()
    
    if return_shapes:
        return loss, batched_shapes
    else:
        return loss

def points(p: torch.Tensor):
    directions = np.array([
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [0, 1],
        [1, 1],
        [1, 0],
        [1, -1],
        [0, -1]
    ])
    new_points = p + directions
    return new_points

def valid_points(points, width, height):
    for x, y in points:
        if x < 0 or x >= width:
            continue
        if y < 0 or y >= height:
            continue
        yield x, y

def get_points(p, width, height):
    new_points = points(p)
    yield from valid_points(new_points, width, height)

def segment(data: torch.Tensor, threshold: float = 0.1):
    batch, _, width, height = data.shape
    
    batched_shapes = []

    for b in range(batch):

        shapes = []
        visited = set()
        stack = [[(0, 0), None]]

        # current_shape = None

        while stack:
            # get the next item from the stack
            current, current_shape = stack.pop()
            x, y = current

            # check if the point has been seen before
            if current in visited:
                # already processed
                continue

            # mark as visited
            visited.add((x, y))

            # get surrounding points
            surrounding = list(get_points(np.array([x, y]), width, height))
            shuffle(surrounding)


            val = data[b, :, x, y]

            if current_shape is None:
                current_shape = [val, [(x, y)]]
                # also add the reference to this shape to the global list
                shapes.append(current_shape)
            else:

                source_val = current_shape[0]

                if torch.abs(val - source_val) > threshold:
                    # print('starting new shape', x, y, source_val.item(), val.item())
                    # start a new shape
                    # current_shape = [val, [(x, y)]]
                    # shapes.append(current_shape)
                    current_shape = None
                else:
                    # add to the current shape
                    current_shape[1].append((x, y))
            
            stack.extend([[s, current_shape] for s in surrounding])

            # stack = [[s, current_shape] for s in surrounding] + stack

        
        batched_shapes.append(shapes)

    return batched_shapes



    
