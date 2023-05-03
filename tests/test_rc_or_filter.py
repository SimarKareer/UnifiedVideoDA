from mmseg.models.uda.dacs import rare_class_or_filter
import torch

def main():
    pl1 = torch.tensor([
        [9, 18, 1, 2],
        [15, 4, 10, 0]
    ])
    
    pl2 = torch.tensor([
        [0, 18, 1, 2],
        [15, 4, 10, 0]
    ])
    ans = torch.tensor([
        [255, 18, 1, 2],
        [15, 4, 10, 0]
    ])
    
    filtered = rare_class_or_filter(pl1, pl2)
    assert(torch.all(filtered == ans))
    

    pl1 = torch.tensor([
        [16, 18, 1, 16],
        [15, 4, 10, 0]
    ])
    
    pl2 = torch.tensor([
        [0, 18, 1, 0],
        [15, 4, 14, 0]
    ])
    ans = torch.tensor([
        [16, 18, 1, 16],
        [15, 4, 255, 0]
    ])
    
    filtered = rare_class_or_filter(pl1, pl2)
    print(filtered)
    assert(torch.all(filtered == ans))

    pl1 = torch.tensor([
        [16, 18, 1, 16],
        [15, 4, 14, 0]
    ])
    
    pl2 = torch.tensor([
        [0, 18, 1, 0],
        [15, 4, 11, 0]
    ])
    ans = torch.tensor([
        [16, 18, 1, 16],
        [15, 4, 11, 0]
    ])
    
    filtered = rare_class_or_filter(pl1, pl2)
    assert(torch.all(filtered == ans))


    pl1 = torch.tensor([
        [16, 18, 1, 16],
        [15, 4, 14, 0]
    ])
    
    pl2 = torch.tensor([
        [0, 18, 1, 0],
        [15, 4, 11, 0]
    ])
    ans = torch.tensor([
        [16, 18, 1, 16],
        [15, 4, 11, 0]
    ])
    
    filtered = rare_class_or_filter(torch.stack([pl1, pl1]), torch.stack([pl2, pl2]))
    assert(torch.all(filtered == torch.stack([ans, ans])))

    

    # print(filtered)


main()