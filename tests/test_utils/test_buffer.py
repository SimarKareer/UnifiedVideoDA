from tools.aggregate_flows.flow.my_utils import CircularTensor
import torch


def main():

    print("Test 1: ")

    buffer = CircularTensor(10)
    buffer.append(torch.ones(9))
    buffer.append(torch.tensor([2]))
    mean = buffer.get_mean()
    print(mean)


    print("Test 2: ")
    buffer = CircularTensor(10)
    buffer.append(torch.ones(9))
    mean = buffer.get_mean()
    print(mean)


    print("Test 3:")

    buffer = CircularTensor(10)
    buffer.append(torch.ones(9))
    buffer.append(torch.tensor([2,3,4,5,6]))
    a = buffer.get_buffer()
    mean = buffer.get_mean()
    print(a)
    print(mean)

    print("Test 4:")

    buffer = CircularTensor(10)
    mean = buffer.get_mean()
    print(mean)

    print("Test 4:")

    buffer = CircularTensor(10)
    buffer.append(torch.ones(10) * 2)
    mean = buffer.get_mean()
    a = buffer.get_buffer()
    print(mean)
    print(a)

    print("Test 5:")

    buffer = CircularTensor(10)
    buffer.append(torch.tensor([1]))
    mean = buffer.get_mean()
    a = buffer.get_buffer()
    print(mean)
    print(a)

    print("Test 6:")
    buffer = CircularTensor(10)
    buffer.append(torch.ones(10) * 2)
    buffer.append(torch.ones(2) * 3)
    mean = buffer.get_mean()
    a = buffer.get_buffer()
    print(mean)
    print(a)

main()