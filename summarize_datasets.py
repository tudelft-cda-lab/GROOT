from groot.datasets import load_all, load_epsilons_dict

epsilons = load_epsilons_dict()
for name, X, y in load_all():
    print(name, *X.shape, epsilons[name], sep="\t")
