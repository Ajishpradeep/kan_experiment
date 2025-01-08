import torch
from kan import KAN
from kan.utils import create_dataset, ex_round
import matplotlib.pyplot as plt
import os

# Set default dtype and device
torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Create a directory to save plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Step 1: Initialize KAN model
# Create a KAN with 2D inputs, 5 hidden neurons, cubic splines (k=3), 3 grid intervals
model = KAN(width=[2, 5, 1], grid=3, k=3, seed=42, device=device)

# Step 2: Create dataset
# Target function: f(x, y) = sin(x) + cos(y) + xy
f = lambda x: torch.sin(x[:, [0]]) + torch.cos(x[:, [1]]) + x[:, [0]] * x[:, [1]]
dataset = create_dataset(f, n_var=2, device=device)
print(dataset["train_input"].shape, dataset["train_label"].shape)


# Helper function to label nodes and edges in the plot
def plot_with_labels(model, title, save_path):
    model.plot()
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


# Step 3: Plot KAN at initialization
model(dataset["train_input"])
plot_with_labels(
    model, "KAN at Initialization", os.path.join(output_dir, "kan_initialization.png")
)

# Step 4: Train KAN with sparsity regularization
model.fit(dataset, opt="LBFGS", steps=50, lamb=0.001)
plot_with_labels(model, "Trained KAN", os.path.join(output_dir, "kan_trained.png"))

# Step 5: Prune KAN and replot
model = model.prune()
plot_with_labels(model, "Pruned KAN", os.path.join(output_dir, "kan_pruned.png"))

# Step 6: Continue training and refine
model.fit(dataset, opt="LBFGS", steps=50)
model = model.refine(10)
model.fit(dataset, opt="LBFGS", steps=50)
plot_with_labels(
    model, "Refined and Retrained KAN", os.path.join(output_dir, "kan_refined.png")
)

# Step 7: Set symbolic activation functions
mode = "auto"  # "manual" for manual configuration
if mode == "manual":
    model.fix_symbolic(0, 0, 0, "sin")
    model.fix_symbolic(0, 1, 0, "cos")
    model.fix_symbolic(1, 0, 0, "x*y")
elif mode == "auto":
    lib = ["x", "x^2", "x^3", "x^4", "exp", "log", "sqrt", "tanh", "sin", "cos", "abs"]
    model.auto_symbolic(lib=lib)

# Step 8: Continue training to machine precision
model.fit(dataset, opt="LBFGS", steps=50)

# Step 9: Obtain symbolic formula
symbolic_formula = model.symbolic_formula()
result = ex_round(symbolic_formula[0][0], 4)
print("Learned Symbolic Formula:", result)
