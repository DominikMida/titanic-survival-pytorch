import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import torch
import pickle
import matplotlib.pyplot as plt
from umtools.datasets import load_stream
stream_np = load_stream()
stream = torch.FloatTensor(stream_np)


img_1 = stream[:2500].reshape(50, 50)
img_2 = stream[2500:12500].reshape(100, 100)
img_3 = stream[12500:].reshape(50, 100, 5)

print(img_1.shape)
print(img_2.shape)
print(img_3.shape)

with open("reshaped.pkl", "wb") as f:
    pickle.dump(img_3, f)

plt.figure(figsize=(20, 10))
plt.subplot(1, 3, 1)
plt.imshow(img_1, cmap="gray")
plt.subplot(1, 3, 2)
plt.imshow(img_2, cmap="gray")
plt.subplot(1, 3, 3)
plt.imshow(img_3[..., :3])
plt.show()


C_rows = img_1
C_columns = img_2

C_rows_inv = torch.inverse(C_rows)
C_cols_inv = torch.inverse(C_columns)

P_enc = img_3

P_enc_permuted = P_enc.permute(2, 0, 1)

P_dec = C_rows_inv @ P_enc_permuted @ C_cols_inv.T
P_dec = P_dec.permute(1, 2, 0)

print(P_dec.shape)

plt.figure(figsize=(10, 5))
plt.imshow(P_dec[..., :3])
plt.show()
with open("unencrypted.pkl", "wb") as f:
    pickle.dump(P_dec, f)


C_rows = img_1
C_columns = img_2
P_enc = img_3

pi_rows = torch.argmax(C_rows, dim=1)
pi_cols = torch.argmax(C_columns, dim=1)

P_enc_chw = P_enc.permute(2, 0, 1)

P_dec_chw = P_enc_chw[..., torch.argsort(pi_rows), :]
P_dec_chw = P_dec_chw[..., torch.argsort(pi_cols)]

P_dec = P_dec_chw.permute(1, 2, 0)
plt.figure(figsize=(10, 5))
plt.imshow(P_dec[..., :3])
plt.show()


from umtools.datasets import load_noise

noise = torch.FloatTensor(load_noise())

print("P_dec shape:", P_dec.shape)
print("noise shape:", noise.shape)

denoised = P_dec - noise.unsqueeze(-1)
print("denoised shape:", denoised.shape)

plt.figure(figsize=(10, 5))
plt.imshow(denoised[..., :3].clamp(0, 1).detach().numpy())
plt.title("Odszumione zdjęcie (RGB)")
plt.show()

with open("denoised.pkl", "wb") as f:
    pickle.dump(denoised, f)


from umtools.utils import save_image

infrared = denoised[..., 3]
print("Infrared channel shape:", infrared.shape)

plt.figure(figsize=(10, 5))
plt.imshow(infrared.detach().numpy(), cmap="inferno")
plt.colorbar()
plt.title("Kanał podczerwieni (IR)")
plt.show()


anomaly_patch = infrared[:20, -20:]

plt.figure(figsize=(4, 4))
plt.imshow(anomaly_patch.detach().numpy(), cmap="gray")
plt.title("Anomalia (IR, 20×20)")
plt.show()

save_image(anomaly_patch.detach().numpy(), "alien.png")


from umtools.functions import Hill

hill = Hill()
Hill().plot()


pos = torch.tensor([1.5, -1.0], requires_grad=True)
lr = 0.01
max_height = float('-inf')

for step in range(1000):
    h = hill(pos)
    h.backward()

    with torch.no_grad():
        pos += lr * pos.grad
        current_h = hill(pos).item()
        if current_h > max_height:
            max_height = current_h

    pos.grad.zero_()


print(max_height)

with open("max_height.pkl", "wb") as f:
    pickle.dump([max_height], f)


import numpy as np
from umtools.functions import Hill
hill = Hill()

x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)

X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

for i in range(200):
    for j in range(200):
        Z[i, j] = hill(torch.tensor([X[i, j], Y[i, j]]))

plt.figure(figsize=(6, 5))
plt.imshow(Z, cmap="terrain", origin="lower")
plt.colorbar(label="wysokość")
plt.title("Hill – mapa wysokości")
plt.show()


results = []
lr = 0.01
n_starts = 10
n_steps = 1000

pos = torch.empty(2, requires_grad=False)

for i in range(n_starts):
    with torch.no_grad():
        pos.uniform_(-5.0, 5.0)

    start_pos = pos.tolist()
    pos.requires_grad_(True)

    with torch.no_grad():
        start_height = hill(pos).item()

    max_h = float('-inf')

    for step in range(n_steps):
        h = hill(pos)
        h.backward()

        with torch.no_grad():
            pos += lr * pos.grad
            current_h = hill(pos).item()
            if current_h > max_h:
                max_h = current_h

        pos.grad.zero_()

    print(f"[{i+1:2d}] start={start_pos}  h_start={start_height:.3f}  h_max={max_h:.3f}")
    results.append((start_pos, start_height, max_h))

    pos = pos.detach()

with open("start_search.pkl", "wb") as f:
    pickle.dump(results, f)

print(max(r[2] for r in results))


import torch.nn as nn
import torch.optim as optim
from umtools.datasets import make_planar

X, y = make_planar()

plt.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10")
plt.title("Zbiór make_planar")
plt.show()

X_t = torch.from_numpy(X).float()
y_t = torch.from_numpy(y).long()


n_features = X_t.shape[1]
n_classes  = len(y_t.unique())

print(f"Cechy: {n_features}, Klasy: {n_classes}, Próbki: {len(y_t)}")


model = nn.Sequential(
    nn.Linear(n_features, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, n_classes),
)

print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters())

n_epochs = 300
loss_history = []

for epoch in range(n_epochs):
    y_pred = model(X_t)
    loss = criterion(y_pred, y_t)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_history.append(loss.item())

plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoka")
plt.ylabel("CrossEntropyLoss")
plt.title("Strata modelu PyTorch")
plt.tight_layout()
plt.savefig("torch_loss.png")
plt.show()

with open("torch_loss.pkl", "wb") as f:
    pickle.dump(loss_history, f)


torch.save(model.state_dict(), "color_model.pt")


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

rf = RandomForestClassifier(
    n_estimators=0,
    warm_start=True,
    random_state=42,
)

trees_per_epoch = 10
n = 30
rf_loss_history = []

for epoch in range(n):
    rf.n_estimators += trees_per_epoch
    rf.fit(X, y)

    proba = rf.predict_proba(X)
    loss_val = log_loss(y, proba)
    rf_loss_history.append(loss_val)



plt.figure(figsize=(8, 4))
plt.plot([i * trees_per_epoch for i in range(1, n + 1)], rf_loss_history)
plt.xlabel("Liczba drzew")
plt.ylabel("log_loss")
plt.title("Strata modelu RandomForest")
plt.tight_layout()
plt.savefig("sklearn_loss.png")
plt.show()



with torch.no_grad():
    preds = model(X_t).argmax(dim=1)
    acc_torch = (preds == y_t).float().mean().item()

acc_rf = (rf.predict(X) == y).mean()

print(f"\nDokładność PyTorch MLP:   {acc_torch*100:.1f}%")
print(f"Dokładność RandomForest:  {acc_rf*100:.1f}%")


import numpy as np
import pandas as pd
import pickle
from umtools.datasets import load_wisla

train_df, test_df = load_wisla()


wiek_median   = train_df['wiek'].median()
port_moda     = train_df['port_wsiadania'].mode()[0]
oplata_median = train_df['oplata'].median()

for df in [train_df, test_df]:
    df['wiek']           = df['wiek'].fillna(wiek_median)
    df['port_wsiadania'] = df['port_wsiadania'].fillna(port_moda)
    df['oplata']         = df['oplata'].fillna(oplata_median)


Q1      = train_df['oplata'].quantile(0.25)
Q3      = train_df['oplata'].quantile(0.75)
granica = Q3 + 1.5 * (Q3 - Q1)
train_df['oplata'] = train_df['oplata'].clip(upper=granica)


drop_cols = ['nr_pasazera', 'sny_przed_podroza', 'ostatni_posilek',
             'wodka_wypita_ml', 'frakcja_majonezowa',
             'cebula_w_bagazu_kg', 'znajomosc_tekstow_disco_polo']
train_df = train_df.drop(columns=drop_cols)
test_df  = test_df.drop(columns=drop_cols)


def extract_tytul(name):
    return name.strip().split()[0]

for df in [train_df, test_df]:
    df['tytul'] = df['pasazer'].apply(extract_tytul)

plec_map = {
    'Pan': 0, 'Panicz': 0, 'Ks.': 0, 'Dr': 0, 'Dr.': 0,
    'Mjr': 0, 'Mjr.': 0, 'Płk': 0, 'Kpt.': 0, 'Hr.': 0,
    'Prof.': 0, 'Prof': 0,
    'Pani': 1, 'Panna': 1,
}
for df in [train_df, test_df]:
    df['plec'] = df['tytul'].map(plec_map).fillna(-1).astype(int)

train_df = train_df.drop(columns=['pasazer'])
test_df  = test_df.drop(columns=['pasazer'])


num_cols  = ['wiek', 'rodzenstwo_malzonek', 'rodzice_dzieci', 'oplata']
num_stats = {col: (train_df[col].mean(), train_df[col].std()) for col in num_cols}

for col in num_cols:
    mean, std = num_stats[col]
    train_df[col] = (train_df[col] - mean) / std
    test_df[col]  = (test_df[col]  - mean) / std


for col in ['port_wsiadania', 'strona_burty', 'tytul']:
    train_df = pd.get_dummies(train_df, columns=[col], prefix=col)
    test_df  = pd.get_dummies(test_df,  columns=[col], prefix=col)

train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)

train_df = train_df.astype({col: int for col in train_df.select_dtypes('bool').columns})
test_df  = test_df.astype({col: int for col in test_df.select_dtypes('bool').columns})

print(train_df.shape)
print(train_df.head())


X_train = train_df.drop(columns=['przezycie']).values.astype(np.float32)
y_train = train_df['przezycie'].values.astype(np.int64)
X_test  = test_df.drop(columns=['przezycie'], errors='ignore').values.astype(np.float32)

print(f"Train: {X_train.shape},  etykiety: {y_train.shape}")
print(f"Test:  {X_test.shape}")


with open("wisla_train.pkl", "wb") as f:
    pickle.dump((X_train, y_train), f)

with open("wisla_test.pkl", "wb") as f:
    pickle.dump(X_test, f)


import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import matplotlib.pyplot as plt


with open("wisla_train.pkl", "rb") as f:
    X_train, y_train = pickle.load(f)

print(f"Trening: {X_train.shape}")


X_t = torch.FloatTensor(X_train)
y_t = torch.LongTensor(y_train)

n_features = X_t.shape[1]


model = nn.Sequential(
    nn.Linear(n_features, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 2),
)

print(model)


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

n_epochs     = 300
loss_history = []

for epoch in range(n_epochs):
    y_pred = model(X_t)
    loss   = criterion(y_pred, y_t)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    loss_history.append(loss.item())


plt.figure(figsize=(8, 4))
plt.plot(loss_history)
plt.xlabel("Epoka")
plt.ylabel("CrossEntropyLoss")
plt.title("Strata modelu – SS Wisła")
plt.tight_layout()
plt.show()


with torch.no_grad():
    preds = model(X_t).argmax(dim=1)
    acc   = (preds == y_t).float().mean().item()

print(f"Dokładność na treningu: {acc*100:.1f}%")


torch.save(model, "wisla.pt")



model.eval()
with torch.no_grad():
    logits = model(X_t)
    loss = criterion(logits, y_t)

print(f"Cross entropy (train): {loss.item():.4f}")
