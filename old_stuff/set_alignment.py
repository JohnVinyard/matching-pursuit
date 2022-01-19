import torch
import numpy as np

def aligned(x, target):
    result = []

    dist = torch.cdist(x, target)
    indices = torch.argsort(dist, dim=-1)

    for batch in indices:
        used = set()
        ind = []
        for row in batch:

            index = 0
            while row[index].item() in used:
                index += 1
                if index >= len(row):
                    break
            
            value = row[index].item()

            used.add(value)
            ind.append(value)
            
        result.append(ind)
    
    result = np.array(result)
    return torch.from_numpy(result)


def reorder(inp, result):

    output = []
    for i, row in enumerate(result):
        output.append(inp[i][result[i]][None, ...])
    
    output = torch.cat(output, dim=0)
    return output



if __name__ == '__main__':
    batch_size = 3
    dim = 2
    time = 5

    inp = torch.FloatTensor(batch_size, time, dim).normal_(0, 1)
    target = torch.FloatTensor(batch_size, time, dim).normal_(0, 1)

    result = aligned(inp, target)
    print(inp)

    print(result)

    print(reorder(inp, result))
