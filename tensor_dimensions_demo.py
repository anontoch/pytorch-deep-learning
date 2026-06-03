import torch



t0 = torch.tensor(5)



t2 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])



t3 = torch.tensor([
    [
        [1, 2],
        [3, 4],
        [5, 6]
    ],
    [
        [7, 8],
        [9, 10],
        [11, 12]
    ]
])


t4 = torch.tensor([
    [
        [
            [1, 2],
            [3, 4],
            [5, 6]
        ],
        [
            [7, 8],
            [9, 10],
            [11, 12]
        ]
    ],
    [
        [
            [13, 14],
            [15, 16],
            [17, 18]
        ],
        [
            [19, 20],
            [21, 22],
            [23, 24]
        ]
    ]
])


print("0D Tensor:", t0)
print("Shape:", t0.shape)

print("\n2D Tensor:\n", t2)
print("Shape:", t2.shape)

print("\n3D Tensor:\n", t3)
print("Shape:", t3.shape)

print("\n4D Tensor:\n", t4)
print("Shape:", t4.shape)