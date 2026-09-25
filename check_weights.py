import torch

models = [
    "weights/best_film_512_effnet.pth",
    "weights/best_film_512_densenet.pth",
    "weights/best_film_512_resnet.pth"
]

for path in models:
    print("\nMODEL:", path)

    checkpoint = torch.load(path, map_location="cpu")

    state_dict = checkpoint["model_state"]

    for key, value in state_dict.items():
        if "tabular" in key:
            print(key, value.shape)