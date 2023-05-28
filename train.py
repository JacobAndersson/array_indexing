import torch
import random
from transformer_lens import HookedTransformer, HookedTransformerConfig 
import pickle
from dataclasses import asdict

str2int = {
    '0': 0,
    '1': 1,
    '2': 2,
    '3': 3,
    '4': 4,
    '5': 5,
    '6': 6,
    '7': 7,
    '8': 8,
    '9': 9,
    ' ': 10,
    '\n': 11,
    '>': 12,
    '=': 13,
    '[': 14,
    ']': 15
}
int2str = {v: k for k, v in str2int.items()}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def gen_list(n, without_rep=True):
    if without_rep:
        return random.sample(range(10), n)
    else:
        return [random.randint(0, 9) for _ in range(n)]

def gen_ex():
    a = gen_list(3)
    idx = random.randint(0, 2)

    y = random.randint(0, 9)
    while y in a:
        y = random.randint(0, 9)

    template = '''{}
[{}]={}
>{}'''

    b = a.copy()
    b[idx] = y

    return template.format("".join(map(str, a)), idx, y, "".join(map(str, b)))

def str_to_tokens(ex):
    return [str2int[c] for c in ex]

def tokens_to_str(tokens):
    return "".join([int2str[t] for t in tokens])

def gen_data(n, batch_size):
    data = []
    used = {}
    while len(data) < n:
        ex = gen_ex()
        curr = str_to_tokens(ex)
        if ex not in used:
            used[ex] = True
            data.append(curr)

    data = torch.tensor(data, dtype=torch.long)
    batches = torch.split(data, batch_size)
    return batches

def criterion(y_pred, y_true):
    pred_tokens = y_pred[:, -4:, :]
    y_true_tokens = y_true[:, -3:]
    probs = torch.log_softmax(pred_tokens, dim=-1)
    correct_probs = torch.gather(probs, 2, y_true_tokens.unsqueeze(-1)).squeeze(-1)
    return -torch.mean(correct_probs)

def save_model(model, cfg, path):
    print('Saving model to', path)
    with open(path, 'wb') as f:
        obj = {
            'model': model.state_dict(),
            'cfg': asdict(cfg),
        }
        pickle.dump(obj, f)

cfg = HookedTransformerConfig(
    d_model=128,
    n_layers=1,
    n_heads=2,
    d_head=64,
    n_ctx=14,
    d_vocab=16,
    act_fn='relu',
    attn_only=True,
    device=device,
    seed=42,
    normalization_type=None,
)

model = HookedTransformer(cfg)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

data = gen_data(7000, 16)
split_idx = int(len(data) * 0.9)
train_data = data[:split_idx]
test_data = data[split_idx:]

for epoch in range(15):
    print('-' * 20)
    print('Epoch', epoch)

    for i, batch in enumerate(train_data):
        batch = batch.to(device)
        out = model(batch)
        loss = criterion(out, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % 100 == 0:
            print(epoch, i, loss.item())

    with torch.no_grad():
        losses = []
        for i, batch in enumerate(test_data):
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch)
            losses.append(loss.item())
        print("###################")
        print('Test loss:', sum(losses) / len(losses))
        print("###################")

    save_model(model, cfg, f'./models/model_1l_{epoch}.pkl')
