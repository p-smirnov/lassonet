import pickle
import numpy as np
import matplotlib.pyplot as plt


lap_path = pickle.load(open("examples/full_batch_grid_search_M/best_model_Lapatinib.p", "rb"))

l3w = [np.sum(np.abs(x.state_dict['layers.3.weight'].numpy())) for x in lap_path[3][0]]
lmbds = [x.lambda_ for x in lap_path[3][0]]


l2w = [np.sum(np.abs(x.state_dict['layers.2.weight'].numpy())) for x in lap_path[3][0]]
l1w = [np.sum(np.abs(x.state_dict['layers.1.weight'].numpy())) for x in lap_path[3][0]]
l0w = [np.sum(np.abs(x.state_dict['layers.0.weight'].numpy())) for x in lap_path[3][0]]

l3b = [np.sum(np.abs(x.state_dict['layers.3.bias'].numpy())) for x in lap_path[3][0]]
l2b = [np.sum(np.abs(x.state_dict['layers.2.bias'].numpy())) for x in lap_path[3][0]]
l1b = [np.sum(np.abs(x.state_dict['layers.1.bias'].numpy())) for x in lap_path[3][0]]
l0b = [np.sum(np.abs(x.state_dict['layers.0.bias'].numpy())) for x in lap_path[3][0]]

plt.subplot(421)
plt.title("Layer 0 weights")
plt.plot(lmbds, l0w)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Sum Absolute Weights")

plt.subplot(423)
plt.title("Layer 1 weights")
plt.plot(lmbds, l1w)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Sum Absolute Weights")

plt.subplot(425)
plt.title("Layer 2 weights")
plt.plot(lmbds, l2w)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Sum Absolute Weights")

plt.subplot(427)
plt.title("Layer 3 weights")
plt.plot(lmbds, l3w)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Sum Absolute Weights")

plt.subplot(422)
plt.title("Layer 0 bias")
plt.plot(lmbds, l1b)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Sum Absolute Bias")

plt.subplot(424)
plt.title("Layer 1 bias")
plt.plot(lmbds, l1b)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Sum Absolute Bias")

plt.subplot(426)
plt.title("Layer 2 bias")
plt.plot(lmbds, l2b)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Sum Absolute Bias")

plt.subplot(428)
plt.title("Layer 3 bias")
plt.plot(lmbds, l3b)
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("Sum Absolute Bias")
plt.tight_layout()

plt.savefig("examples/full_batch_grid_search_M/lapatinib_weight_over_lambda.png", dpi=300)
