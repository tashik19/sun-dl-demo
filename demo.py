import math, torch, torch.nn as nn

torch.manual_seed(0)

def make_spirals(n_per_class=1000, noise=0.15):
    t = torch.linspace(0.25, 3.0, n_per_class) * 2 * math.pi / 1.5
    r = t / (2 * math.pi) * 1.5
    x0 = torch.stack([r * torch.cos(t), r * torch.sin(t)], 1)
    x1 = -x0                                   # second spiral = first rotated 180 degrees
    X = torch.cat([x0, x1]) + noise * torch.randn(2 * n_per_class, 2)
    y = torch.cat([torch.zeros(n_per_class), torch.ones(n_per_class)])
    return X, y

X, y = make_spirals()
perm = torch.randperm(len(X))
X, y = X[perm], y[perm]
split = int(0.8 * len(X))
Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]

model = nn.Sequential(
    nn.Linear(2, 64), nn.ReLU(),
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 1),
)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

def accuracy(Xs, ys):
    with torch.no_grad():
        pred = (model(Xs).squeeze(1) > 0).float()
    return (pred == ys).float().mean().item()

print(f"params: {sum(p.numel() for p in model.parameters())}  train={len(Xtr)} test={len(Xte)}")
print(f"before training: test acc = {accuracy(Xte, yte):.3f}")

for epoch in range(1, 301):
    opt.zero_grad()
    loss = loss_fn(model(Xtr).squeeze(1), ytr)
    loss.backward()
    opt.step()
    if epoch % 50 == 0 or epoch == 1:
        print(f"epoch {epoch:3d}  loss={loss.item():.4f}  train acc={accuracy(Xtr, ytr):.3f}  test acc={accuracy(Xte, yte):.3f}")

print(f"FINAL test accuracy: {accuracy(Xte, yte):.3f}")
