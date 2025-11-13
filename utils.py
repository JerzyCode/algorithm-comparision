#
# with open("experiment_result/Sphere_2025-11-13.json", "r") as f:
#     data = json.load(f)
#
# for optim_name, optim_data in data["optimizers"].items():
#     print(f"{optim_name} params:", optim_data["params"])
#     values = np.array(optim_data["values"])
#     mean = values.mean(axis=0)
#     std = values.std(axis=0)
#     print(f"{optim_name} mean loss:", mean)
