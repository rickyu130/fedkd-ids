from __future__ import annotations
import copy, math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

def set_seed(seed, deterministic=False):
    """Seed the torch RNGs so a full run is reproducible end to end.

    ``load_mini`` and ``partition`` already use their own seeded
    ``np.random.default_rng``, so the data split was reproducible; this covers the
    two torch sources that were previously unseeded -- model initialisation and the
    shuffled ``DataLoader`` inside ``train``.  Call it once at the start of a runner.

    ``deterministic=True`` additionally forces cuDNN to use deterministic kernels,
    which makes CUDA runs bit-exact at the cost of speed.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed

def resolve_device(device='cuda'):
    """Validate ``device``; raise if CUDA was requested but is unavailable.

    ``cuda`` is the default in every runner so the paper-scale commands do not have
    to spell it out.  Failing loudly beats silently falling back to CPU: a fallback
    would let a multi-hour run look healthy while actually grinding along on CPU.
    Pass ``--device cpu`` explicitly to run on CPU.
    """
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'--device {device} was requested but CUDA is not available. Pass --device cpu to run on CPU.')
    return device

def partition(x,y,devices,scenario=1,seed=7,alpha=.1):
    rng=np.random.default_rng(seed); clients=[]
    for dev in np.unique(devices):
        ids=np.where(devices==dev)[0]; k=3 if scenario==1 else len(np.unique(y[ids]))
        if scenario in (1,2):
            ids=ids[np.argsort(y[ids])]; shards=np.array_split(ids,2*k); rng.shuffle(shards)
            clients += [np.concatenate(shards[i:i+2]) for i in range(0,len(shards),2)]
        else:
            by=[ids[y[ids]==c] for c in np.unique(y[ids])]; buckets=[[] for _ in range(k)]
            for c,arr in enumerate(by):
                # np.cumsum can land a ULP below 1.0, truncating the last cut and dropping the tail.
                rng.shuffle(arr); prop=rng.dirichlet(np.full(k,alpha)); cuts=(np.cumsum(prop)*len(arr)).astype(int); cuts[-1]=len(arr); start=0
                for j,end in enumerate(cuts): buckets[j].extend(arr[start:end]); start=end
            clients += [np.asarray(b,dtype=int) for b in buckets]
    return clients


def partition_fedkd1(x, y, devices, seed=7):
    """Create 27 deterministic, label-skewed Partition 1 collaborators.

    The paper specifies 27 collaborators whose local label support contains
    2--7 classes.  The paper does not publish the exact sample-to-collaborator
    manifest, so we use a reproducible global label-skew construction: each
    collaborator receives two or three labels and every label's samples are
    distributed exactly once across its eligible collaborators.  Device IDs are
    retained in the dataset for auditing but are not used as client boundaries.
    """
    del devices  # Partition 1 is defined over the global private set.
    y = np.asarray(y)
    labels = np.unique(y)
    if len(labels) < 2:
        raise ValueError('FedKD Partition 1 requires at least two labels')
    rng = np.random.default_rng(seed)
    label_order = labels[rng.permutation(len(labels))]
    client_labels = []
    for client_id in range(27):
        a = label_order[client_id % len(label_order)]
        b = label_order[(client_id + 1 + client_id // len(label_order)) % len(label_order)]
        if a == b:
            b = label_order[(client_id + 2) % len(label_order)]
        support = [a, b]
        if client_id % 5 == 0 and len(label_order) >= 3:
            c = label_order[(client_id + 2 + client_id // len(label_order)) % len(label_order)]
            if c not in support:
                support.append(c)
        client_labels.append(support)

    client_ids = [[] for _ in client_labels]
    for label in labels:
        eligible = [i for i, support in enumerate(client_labels) if label in support]
        sample_ids = np.flatnonzero(y == label)
        sample_ids = sample_ids[rng.permutation(len(sample_ids))]
        for client_id, chunk in zip(eligible, np.array_split(sample_ids, len(eligible))):
            client_ids[client_id].extend(chunk.tolist())
    out = []
    for ids in client_ids:
        ids = np.asarray(ids, dtype=int)
        out.append(ids[rng.permutation(len(ids))])
    return out


def partition_partition1(y, seed=7, population_counts=None, clients=27,
                         private_ratio=0.70, shares_per_label=8):
    """Allocate Partition 1's full private set under the paper's per-label rule.

    Each label is assigned to eight collaborators. Since private training is
    70% of the full per-label population, each equal shard represents 8.75%
    of that population, satisfying the paper's 6--10% collaborator constraint.
    The deterministic cyclic incidence matrix gives every collaborator 2--7
    classes and every private sample is assigned exactly once.
    """
    y = np.asarray(y)
    labels = np.unique(y)
    if clients != 27 or shares_per_label != 8:
        raise ValueError('Partition 1 expects 27 collaborators and 8 shares per label')
    rng = np.random.default_rng(seed)
    label_order = labels[rng.permutation(len(labels))]
    # The seven-step cyclic schedule spreads class support across all clients.
    client_ids = [[] for _ in range(clients)]
    client_labels = [[] for _ in range(clients)]
    for label_idx, label in enumerate(label_order):
        start = (label_idx * 7) % clients
        eligible = [(start + j) % clients for j in range(shares_per_label)]
        ids = np.flatnonzero(y == label)
        ids = ids[rng.permutation(len(ids))]
        chunks = np.array_split(ids, shares_per_label)
        if population_counts is None:
            full_population = len(ids) / private_ratio
        else:
            full_population = float(population_counts[int(label)])
        for client_id, chunk in zip(eligible, chunks):
            share = len(chunk) / full_population
            if len(chunk) and not 0.06 <= share <= 0.10:
                raise ValueError(f'label {label} collaborator share {share:.4f} outside 6--10%')
            client_ids[client_id].extend(chunk.tolist())
            client_labels[client_id].extend([int(label)] * len(chunk))
    result = []
    for ids, local_y in zip(client_ids, client_labels):
        ids = np.asarray(ids, dtype=np.int64)
        local_y = np.asarray(local_y, dtype=np.int64)
        order = rng.permutation(len(ids))
        result.append((ids[order], local_y[order]))
    support = [len(np.unique(local_y)) for _ids, local_y in result]
    if any(count < 2 or count > 7 for count in support):
        raise ValueError(f'Partition 1 client class support outside 2--7: {support}')
    return result

class CNN(nn.Module):
    """Paper's Table I CNN baseline for an ``(N, 23, 5)`` traffic matrix.

    The 23 feature rows are the Conv1D channels and the five time windows are
    the sequence axis.  Layers 7 and 8 use stride 2, reducing the sequence
    length from 5 to 3 and then 2; the classifier therefore consumes exactly
    ``128 * 2 == 256`` features, followed by the paper's 128-unit layer.
    """
    def __init__(self, classes=11, out=11, in_channels=23):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(128, 128, 3, stride=2, padding=1), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.Linear(128, out))

    def forward(self, x):
        return self.head(self.features(x))

class MLP(nn.Module):
    def __init__(self, classes=11, out=None):
        out = classes if out is None else out
        super().__init__(); self.net=nn.Sequential(nn.Flatten(),nn.Linear(115,256),nn.ReLU(),nn.Linear(256,out))
    def forward(self,x): return self.net(x)

class LSTM(nn.Module):
    def __init__(self, classes=11, out=None):
        out = classes if out is None else out
        super().__init__(); self.r=nn.LSTM(5,128,batch_first=True); self.fc=nn.Linear(128,out)
    def forward(self,x): return self.fc(self.r(x)[0][:,-1])

class Discriminator(CNN):
    """CNN discriminator; architecture is shared with the classifier, 2 outputs."""
    def __init__(self, in_channels=23): super().__init__(classes=11, out=2, in_channels=in_channels)

def build_model(name='cnn', classes=11, discriminator=False, in_channels=23):
    """Construct one of the paper's classifier/discriminator architectures."""
    key = name.lower()
    if key == 'cnn':
        return Discriminator(in_channels=in_channels) if discriminator else CNN(classes=classes, in_channels=in_channels)
    out = 2 if discriminator else classes
    if key == 'mlp': return MLP(classes=classes, out=out)
    if key == 'lstm': return LSTM(classes=classes, out=out)
    raise ValueError(f'unknown model architecture: {name}')

def train(model,x,y,epochs=5,lr=1e-4,batch=80,device='cpu',opt=None):
    """Supervised training; ``opt`` lets the caller reuse an Adam across rounds.

    Adam's first step from a zero state moves every parameter by a full ``lr``:
    the bias correction makes ``mhat/sqrt(vhat)`` exactly +-1, so the step is
    ``+-alpha`` regardless of gradient magnitude.  It then takes several steps for
    the moments to settle back to the steady-state signal-to-noise ratio, which is
    smaller.  A client whose classifier persists across rounds should therefore keep
    its optimiser too -- pass back the optimiser returned by the previous call.

    Returns ``(model, opt)``.
    """
    if len(x)==0: return model, opt
    model.to(device).train()
    if opt is None: opt=torch.optim.Adam(model.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss(); dl=DataLoader(TensorDataset(torch.tensor(x),torch.tensor(y,dtype=torch.long)),batch_size=batch,shuffle=True)
    for _ in range(epochs):
        for a,b in dl: opt.zero_grad(); loss(model(a.to(device)),b.to(device)).backward(); opt.step()
    return model, opt

def train_soft(model,x,target,y=None,epochs=5,lr=1e-4,batch=80,device='cpu',distill_weight=1.,opt=None):
    """Train against a soft target distribution, optionally with a supervised term.

    ``target`` is a per-sample distribution (already normalised).  FD and DS-FL
    supervise their models with shared soft-labels rather than hard labels, so the
    distillation loss is the cross-entropy against that distribution,
    ``-sum_l target_l * log_softmax(model(x))_l``.  When ``y`` is given the loss
    follows FD's Eq (7), ``CE(hard) + distill_weight * CE(soft)`` -- the
    ground-truth term plus a weighted distillation regulariser.  DS-FL's step 5 is
    pure distillation (Eq 10), so it passes ``y=None``.

    ``opt`` reuses an Adam across rounds, same rationale as in ``train``.
    Returns ``(model, opt)``.
    """
    if len(x)==0: return model, opt
    model.to(device).train()
    if opt is None: opt=torch.optim.Adam(model.parameters(),lr=lr)
    ts=torch.tensor(target,dtype=torch.float32); xs=torch.tensor(x)
    ds=TensorDataset(xs,ts) if y is None else TensorDataset(xs,ts,torch.tensor(np.asarray(y),dtype=torch.long))
    dl=DataLoader(ds,batch_size=batch,shuffle=True)
    for _ in range(epochs):
        for batch_ in dl:
            a=torch.as_tensor(batch_[0]).to(device); b=torch.as_tensor(batch_[1]).to(device)
            opt.zero_grad(); logp=torch.log_softmax(model(a),1)
            loss=-(b*logp).sum(1).mean()                                          # distillation term
            if y is not None:
                c=torch.as_tensor(batch_[2]).to(device)
                loss=(-logp.gather(1,c.unsqueeze(1)).squeeze(1)).mean()+distill_weight*loss
            loss.backward(); opt.step()
    return model, opt

def to_probs(logits):
    """Softmax over the last axis; turns raw model output into a distribution."""
    z=np.asarray(logits,dtype=np.float32); z=z-z.max(axis=-1,keepdims=True)
    e=np.exp(z); return e/e.sum(axis=-1,keepdims=True)

def normalize_target(t):
    """Clip negative entries and renormalise a leave-one-out target.

    FD's Eq (6) is a linear combination of averaged soft-labels, so it already sums
    to 1 but may contain small negative entries when one client dominates a class.
    Clipping them keeps the cross-entropy well defined.
    """
    t=np.maximum(np.asarray(t,dtype=np.float32),0.0); s=t.sum(-1,keepdims=True)
    return np.divide(t,s,out=np.full_like(t,1.0/t.shape[-1]),where=s>0)

def sharpen(soft_labels,temperature=.1):
    """Entropy Reduction Aggregation (ERA) of DS-FL, Eq (7)-(8).

    The server averages the clients' soft-labels and applies a temperature softmax
    to that average; ``temperature < 1`` lowers the entropy of the consensus, which
    is what keeps DS-FL from stagnating on noisy local predictions.  Note the input
    is the *averaged soft-label* (a probability vector), not a raw logit: DS-FL
    exchanges normalised soft-labels, and the original paper's "logits" is the
    output of the softmax.
    """
    return to_probs(np.asarray(soft_labels,dtype=np.float32)/float(temperature))

def build_discriminator_like(classifier):
    """Build a fresh discriminator matching ``classifier``'s architecture.

    Shared by :func:`discriminator_labels` and checkpoint restore, which needs to
    rebuild a discriminator before it can load the saved weights into it.
    """
    if isinstance(classifier, MLP): disc = MLP(classes=2)
    elif isinstance(classifier, LSTM): disc = LSTM(classes=2)
    else: disc = Discriminator()
    disc(torch.zeros(1,23,5))   # materialise any lazy parameters
    return disc


def predict_proba_eval(model, x, device='cpu'):
    """Predict with Dropout/BatchNorm disabled and restore the prior mode."""
    was_training = model.training
    model.to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.as_tensor(x, dtype=torch.float32).to(device)), 1)
    if was_training:
        model.train()
    return probs.cpu().numpy()

def discriminator_labels(classifier, x_private, x_open, threshold, epochs=5, lr=1e-4, batch=80, device='cpu', disc=None, disc_opt=None):
    """Train paper's familiar/unfamiliar discriminator and return familiar mask.

    ``disc`` is the caller's discriminator from the previous round.  Passing it in
    **warm-starts the discriminator across communication rounds** instead of
    rebuilding it from scratch.  This is what the paper prescribes: Algorithm 1
    step 2 says "train wk,d with Dk,d" -- the same verb used for the classifier in
    step 1 -- and the symbol table states that "each client **has** a discriminator
    wk,d", i.e. it is part of the client's persistent state exactly like wk,c.
    A discriminator rebuilt every round only ever sees ``local_epochs`` rounds of
    training from a random initialisation, which is too weak for the component the
    paper calls the largest single contributor to detection performance.

    Leave ``disc=None`` to build a fresh discriminator (first round, or callers
    that do not keep client state).  ``disc_opt`` is that discriminator's Adam;
    pass back the optimiser returned by the previous call so its moments survive
    too.  Returns ``(familiar_mask, disc, disc_opt)``.
    """
    q = predict_proba_eval(classifier, x_open, device)
    conf=q.max(1); unfamiliar=x_open[conf < threshold]
    xd=np.concatenate([x_private, unfamiliar]); yd=np.concatenate([np.zeros(len(x_private),dtype=np.int64), np.ones(len(unfamiliar),dtype=np.int64)])
    if disc is None:
        disc = build_discriminator_like(classifier)
    disc,disc_opt=train(disc,xd,yd,epochs,lr,batch,device,opt=disc_opt)
    pred = predict_proba_eval(disc, x_open, device).argmax(1)
    return pred == 0, disc, disc_opt

def client_skip_reason(n_private, n_unfamiliar=None):
    """Why client k cannot produce a usable upload this round; ``None`` if it can.

    Eq (13) and Eq (14) build D_k,d out of two halves -- the open samples the
    classifier is unsure about ("unfamiliar") plus the client's whole private set
    ("familiar") -- and step 2 then trains a two-class discriminator on it.  If
    either half is empty, D_k,d is a one-class problem, so Eq (15)'s arg max is
    arbitrary and the client's upload would mean nothing:

    * ``n_private == 0`` -- no "familiar" half, because Eq (14) adds nothing.  The
      Dirichlet partition of Scenario 3 can hand a client zero samples.
    * ``n_unfamiliar == 0`` -- no "unfamiliar" half, because Eq (13) adds nothing.
      This is what a collapsed classifier looks like: identical confidence on every
      open sample, so nothing falls below the threshold.  The reference
      implementation tracks this exact case with its ``sure_unknown_none`` set.

    These are not crashes -- ``train`` already no-ops on an empty batch, so nothing
    goes NaN.  The point is that the upload would be noise, so the client abstains.

    ``n_unfamiliar`` is optional so a caller can screen a client before it has run
    the classifier (pass it once the confidence scores are known).

    Deliberately absent: a rule skipping clients that hold a single private class.
    The reference implementation has one, but the paper never mentions it, and
    Scenario 3 exists precisely to be extreme -- removing its most skewed clients
    would quietly make the scenario easier than the one the paper reports on.
    """
    if n_private == 0: return 'no private samples'
    if n_unfamiliar is not None and n_unfamiliar == 0: return 'no unfamiliar samples for D_k,d'
    return None

def _record_skip(stats, k, reason):
    """Append ``(client index, reason)`` to ``stats['skipped']`` if a caller asked for it."""
    if stats is not None: stats.setdefault('skipped', []).append((k, reason))

class ClientState:
    """Persistent per-client state that survives across communication rounds.

    Algorithm 1 keeps both the classifier (wk,c) and the discriminator (wk,d) on
    the client and keeps saying "train" for both -- so their Adam moments belong to
    the same persistent state as the weights themselves.  Rebuilding the optimiser
    every round throws away the accumulated gradient statistics and restarts each
    round with a full-``lr`` first step (see the note in ``train``).
    """
    __slots__ = ('disc', 'opt', 'disc_opt')
    def __init__(self, disc=None, opt=None, disc_opt=None):
        self.disc = disc          # SSFL discriminator; stays None for other methods
        self.opt = opt            # classifier optimiser
        self.disc_opt = disc_opt  # discriminator optimiser

def save_checkpoint(path, method, rnd, server, server_opt, client_models, client_states):
    """Persist everything the next round needs, so a long run can resume after a crash.

    ``rnd`` is the number of rounds **completed** for ``method``; a resume picks up
    at round ``rnd`` (0-based).  The client classifiers, their discriminators and
    every Adam ``state_dict`` are included: without the optimiser moments a resumed
    run would restart each round with a full-``lr`` first step and would not match
    an uninterrupted one (see the note in ``train``).  The torch RNG states go in
    too, because ``train`` shuffles through the global RNG -- without them the
    resumed run would draw a different batch order and diverge from the original.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'method': method,
        'round': rnd,
        'torch_rng': torch.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'server': None if server is None else server.state_dict(),
        'server_opt': None if server_opt is None else server_opt.state_dict(),
        'clients': None if client_models is None else [m.state_dict() for m in client_models],
        'client_states': None if client_states is None else [
            {'disc': None if st.disc is None else st.disc.state_dict(),
             'opt': None if st.opt is None else st.opt.state_dict(),
             'disc_opt': None if st.disc_opt is None else st.disc_opt.state_dict()}
            for st in client_states],
    }, path)

def load_checkpoint(path):
    """Load a checkpoint written by :func:`save_checkpoint`; ``None`` if it is absent."""
    p = Path(path)
    if not p.exists(): return None
    return torch.load(p, map_location='cpu', weights_only=False)

def restore_checkpoint(ckpt, server, client_models, client_states, lr=1e-4, device=None):
    """Load ``ckpt`` into the given live models, rebuilding optimisers as needed.

    An Adam ``state_dict`` is keyed by parameter *position*, not identity, so the
    optimiser must be reconstructed from the model's parameters before its state can
    be restored.  Client discriminators are rebuilt from their classifier's
    architecture when the checkpoint holds one.  Models are mutated in place; the
    torch RNG states are restored so the resumed batch order matches the original.
    ``device`` selects the training device. Move models there BEFORE rebuilding
    Adam, so loading its state places moment tensors alongside the parameters.
    Leave it unset to keep the supplied models on their current devices.
    Returns ``(server_opt, client_states)``.
    """
    if ckpt is None: return None, client_states
    if device is not None:
        if server is not None: server.to(device)
        if client_models is not None:
            for model in client_models: model.to(device)
    if ckpt.get('server') is not None and server is not None:
        server.load_state_dict(ckpt['server'])
    server_opt = None
    if ckpt.get('server_opt') is not None and server is not None:
        server_opt = torch.optim.Adam(server.parameters(), lr=lr)
        server_opt.load_state_dict(ckpt['server_opt'])
    if ckpt.get('clients') is not None and client_models is not None:
        for m, sd in zip(client_models, ckpt['clients']): m.load_state_dict(sd)
    saved = ckpt.get('client_states')
    if saved is not None and client_states is not None:
        for k, (st, s) in enumerate(zip(client_states, saved)):
            if s['disc'] is not None:
                if st.disc is None: st.disc = build_discriminator_like(client_models[k])
                disc_device = device if device is not None else next(client_models[k].parameters()).device
                st.disc.to(disc_device)
                st.disc.load_state_dict(s['disc'])
            if s['opt'] is not None:
                if st.opt is None: st.opt = torch.optim.Adam(client_models[k].parameters(), lr=lr)
                st.opt.load_state_dict(s['opt'])
            if s['disc_opt'] is not None and st.disc is not None:
                if st.disc_opt is None: st.disc_opt = torch.optim.Adam(st.disc.parameters(), lr=lr)
                st.disc_opt.load_state_dict(s['disc_opt'])
    # Restore the RNG states LAST.  Rebuilding a discriminator runs a forward pass
    # through a module that is still in train mode, so its dropout draws from the
    # global RNG -- doing this before the RNG restore would clobber it and the
    # resumed run would shuffle its batches differently from the original.
    if ckpt.get('torch_rng') is not None: torch.set_rng_state(ckpt['torch_rng'])
    if ckpt.get('cuda_rng') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
    return server_opt, client_states

def fedavg(models, weights=None):
    """Parameter averaging baseline (FL)."""
    if not models: raise ValueError('models must be non-empty')
    weights=np.ones(len(models))/len(models) if weights is None else np.asarray(weights)/sum(weights)
    state=copy.deepcopy(models[0].state_dict())
    for k in state:
        vals=[m.state_dict()[k] for m in models]
        state[k]=sum(float(weights[i])*vals[i] for i in range(len(models))) if vals[0].is_floating_point() else vals[0]
    out=copy.deepcopy(models[0]); out.load_state_dict(state); return out

def class_average_output(outputs, labels, classes=11):
    """Per-class average of the model output, i.e. FD's Eq (4).

    ``outputs`` are the model's softmax probabilities (FD exchanges the inferred
    class probability), so row ``c`` is the local-average soft-label for class
    ``c``.  A class the client holds no sample of gives an all-zero row, exactly
    as the paper specifies.
    """
    out=np.zeros((classes,classes),np.float32); labels=np.asarray(labels)
    for c in range(classes):
        z=outputs[labels==c]; out[c]=z.mean(0) if len(z) else 0
    return out

def evaluate(model,x,y,device='cpu'):
    model.to(device).eval()
    with torch.no_grad(): p=model(torch.tensor(x).to(device)).argmax(1).cpu().numpy()
    return metrics(y,p), p

def federated_round(base, x, y, clients, method='fl', open_x=None, local_epochs=5, lr=1e-4, batch=80, device='cpu', client_models=None, temperature=.1, fd_distill_weight=1., client_states=None, server_opt=None, dist_rounds=None, discri_rounds=None, stats=None):
    """Run one communication round and return ``(server, server_opt, client_models, client_states)``.

    ``client_states`` is a list of :class:`ClientState`, one per client, mirroring
    ``client_models``.  SSFL's discriminator is part of the client's persistent
    state (Algorithm 1 step 2, "each client has a discriminator wk,d"), so it is
    carried across rounds and warm-started rather than rebuilt; the Adam optimisers
    for both the classifier and that discriminator are carried for the same reason.
    Pass ``None`` on the first round.  ``server_opt`` is the server model's own
    Adam, carried for the same reason.

    The server model is trained **in place** (``server is base``) rather than
    deep-copied, because an optimiser holds references to its parameters: copying
    the model every round would leave the carried optimiser pointing at stale
    parameters.  Callers already rebind ``model`` to the returned server.

    ``discri_rounds`` and ``dist_rounds`` give Algorithm 1's discriminator step (2)
    and distillation step (5) their own epoch budgets, independent of the local
    supervised budget ``local_epochs`` (step 1).  The reference implementation uses
    **3** and **10** respectively against a local budget of 3 -- distillation gets
    more than three times the supervised training.  Both default to ``None``, which
    falls back to ``local_epochs`` so existing configurations keep their behaviour;
    set them explicitly to reproduce the reference budget.

    ``stats`` is an optional dict; pass one to collect ``stats['skipped']``, a list
    of ``(client index, reason)`` for clients that could not produce an upload this
    round (see :func:`client_skip_reason`).  A client that is skipped simply does
    not vote; if nobody can upload, the round is a no-op and the classifiers carry
    over unchanged.

    Every distillation method follows its own paper's flow, and all three keep one
    classifier per client across rounds (``client_models``); no model parameter is
    ever aggregated or broadcast, which is why the per-round cost does not scale
    with the model size (paper Section V-E).  Pass ``client_models=None`` on the
    first round to initialize every client from ``base`` (paper: ``wk <- ws``).

    ``ssfl`` -- Zhao et al., Algorithm 1: clients exchange only the voted global
                hard label and distil it into their own classifier (18).
    ``fd``   -- Jeong et al., Section III-B: clients upload per-class average
                soft-labels (4), the server averages them over the clients that hold
                that class (5), each client removes its own contribution (6) and
                trains with its ground-truth labels plus a distillation term (7).
                FD has no server-side classifier, so ``server_model`` is ``None``
                and the caller evaluates the clients.
    ``dsfl`` -- Itahara et al., Section III-C: clients upload their soft-labels on
                the open set (5), the server runs ERA -- average (6) then a
                temperature softmax (7)-(8) -- and every client distils the global
                soft-labels (9) while the server trains its own model (10).
    ``fl``   -- plain FedAvg, for which parameter averaging plus the broadcast of
                the new global model *is* the scheme, so ``client_models`` is
                ``None`` and the clients restart from ``server_model`` each round.

    ``temperature`` is DS-FL's ERA temperature (8), fixed at **0.1** by default.
    ``fd_distill_weight`` is FD's gamma, the weight of the distillation regulariser
    in Eq (7); the paper does not state its value, so it defaults to 1.
    """
    if method=='fl':
        locals_=[]
        for ids in clients:
            m=copy.deepcopy(base); m,_=train(m,x[ids],y[ids],local_epochs,lr,batch,device); locals_.append(m)
        return fedavg(locals_, weights=[len(ids) for ids in clients]), None, None, None
    if open_x is None: raise ValueError('open_x required for distillation methods')

    if method=='ssfl':
        if client_models is None: client_models=[copy.deepcopy(base) for _ in clients]
        if client_states is None: client_states=[ClientState() for _ in clients]
        d_rounds=local_epochs if dist_rounds is None else dist_rounds
        c_rounds=local_epochs if discri_rounds is None else discri_rounds
        outputs=[]
        for k,(m,ids) in enumerate(zip(client_models,clients)):
            reason=client_skip_reason(len(ids))
            if reason: _record_skip(stats,k,reason); continue
            m,opt=train(m,x[ids],y[ids],local_epochs,lr,batch,device,opt=client_states[k].opt)   # (11)
            client_states[k].opt=opt
            outputs.append((k,m,ids,predict_proba_eval(m, open_x, device)))
        filtered=[]
        for k,m,ids,q in outputs:
            conf=q.max(1); threshold=float(np.median(conf))                            # (12)
            reason=client_skip_reason(len(ids),int((conf<threshold).sum()))
            if reason: _record_skip(stats,k,reason); continue
            # (13)-(15): warm-start this client's own discriminator -- and its Adam --
            # from the previous round.
            st=client_states[k]
            familiar,disc,disc_opt=discriminator_labels(m,x[ids],open_x,threshold,epochs=c_rounds,lr=lr,batch=batch,device=device,disc=st.disc,disc_opt=st.disc_opt)
            st.disc=disc; st.disc_opt=disc_opt
            filtered.append(np.where(familiar,q.argmax(1),-1))                        # (16)
        # With no uploads at all the round is a no-op: every sample stays unknown, so
        # there is nothing to distil and the classifiers simply carry over.
        pseudo=vote(filtered) if filtered else np.full(len(open_x),-1,dtype=int)      # (17)
        valid=pseudo>=0
        if valid.any():
            # (18): every client distils the voted global hard labels into its own
            # classifier.  This is the only channel through which a client benefits
            # from the federation -- no parameter ever moves.  Same optimiser as the
            # private step above, matching the paper's single wk,c trained by (11)+(18).
            for k,m in enumerate(client_models):
                m,opt=train(m,open_x[valid],pseudo[valid],d_rounds,lr,batch,device,opt=client_states[k].opt)
                client_states[k].opt=opt
        server=base                                                                   # server keeps its own classifier, trained in place
        if valid.any(): server,server_opt=train(server,open_x[valid],pseudo[valid],d_rounds,lr,batch,device,opt=server_opt)
        return server, server_opt, client_models, client_states

    # ---- FD (Jeong et al.) and DS-FL (Itahara et al.) baselines ----
    # Like SSFL, both keep one classifier per client across rounds and exchange
    # only logits / logit summaries -- never model parameters.  Neither builds a
    # discriminator, so the ``disc`` slots stay None; the optimisers are still
    # carried so the classifiers keep their Adam moments.
    if client_models is None: client_models=[copy.deepcopy(base) for _ in clients]
    if client_states is None: client_states=[ClientState() for _ in clients]

    if method=='fd':
        # Steps 1-2: train locally, then summarise each model by its per-class
        # average soft-label (Eq 4).  FD exchanges the inferred class probability,
        # so the summary is built from the softmax output, not from raw logits; a
        # class the client holds no sample of gives an all-zero row.
        summaries=[]; present=[]
        for k,(m,ids) in enumerate(zip(client_models,clients)):
            # (1): reuse the client's carried Adam, exactly as SSFL and DS-FL do.
            # Rebuilding it here would restart every round with a full-lr first step
            # and would contradict this branch's promise that the optimisers are
            # carried (see the comment above).
            m,opt=train(m,x[ids],y[ids],local_epochs,lr,batch,device,opt=client_states[k].opt)
            client_states[k].opt=opt
            with torch.no_grad(): probs=torch.softmax(m(torch.tensor(x[ids]).to(device)),1).cpu().numpy()
            summaries.append(class_average_output(probs,y[ids],probs.shape[1]))       # (4)
            present.append(np.array([(y[ids]==c).any() for c in range(probs.shape[1])]))
        # Steps 3-4: upload, then the server averages per class over K_nl -- only the
        # clients that actually hold that class (Eq 5).  Clients missing a class must
        # not contribute their zero row.
        S=np.stack(summaries); P=np.stack(present); counts=P.sum(0)
        global_summary=np.divide((S*P[:,:,None]).sum(0),counts[:,None],out=np.zeros_like(S[0]),where=counts[:,None]>0)
        # Steps 5-6: each client removes its own contribution over K_nl (Eq 6) and
        # trains with its ground-truth labels plus the distillation term (Eq 7).
        for k,(m,ids,own,mask) in enumerate(zip(client_models,clients,summaries,present)):
            n=counts[mask][:,None]
            refined=np.zeros_like(own)
            refined[mask]=(n*global_summary[mask]-own[mask])/np.maximum(n-1,1)        # (6)
            m,opt=train_soft(m,x[ids],normalize_target(refined)[y[ids]],y=y[ids],epochs=local_epochs,lr=lr,batch=batch,device=device,distill_weight=fd_distill_weight,opt=client_states[k].opt)
            client_states[k].opt=opt
        # FD has no server-side classifier (Section III-B), so the caller evaluates
        # the clients themselves.
        return None, None, client_models, client_states

    if method=='dsfl':
        # Steps 1-2: train locally, then produce the local soft-labels on the open
        # set (5).  DS-FL exchanges normalised soft-labels -- the original paper's
        # "logits" is the output of the softmax, not a raw logit.
        soft_labels=[]
        for k,(m,ids) in enumerate(zip(client_models,clients)):
            m,opt=train(m,x[ids],y[ids],local_epochs,lr,batch,device,opt=client_states[k].opt)  # (1)
            client_states[k].opt=opt
            with torch.no_grad(): soft_labels.append(torch.softmax(m(torch.tensor(open_x).to(device)),1).cpu().numpy())
        # Steps 3-4: ERA -- the server averages the local soft-labels (6) and applies
        # a temperature softmax to reduce the entropy of the consensus (7)-(8).
        global_probs=sharpen(np.mean(np.stack(soft_labels),axis=0),temperature)
        # Step 5: broadcast; every client distils the global soft-labels (9).
        for k,m in enumerate(client_models):
            m,opt=train_soft(m,open_x,global_probs,epochs=local_epochs,lr=lr,batch=batch,device=device,opt=client_states[k].opt)
            client_states[k].opt=opt
        # Step 6: the server also trains its own global model (10).
        server=base
        server,server_opt=train_soft(server,open_x,global_probs,epochs=local_epochs,lr=lr,batch=batch,device=device,opt=server_opt)
        return server, server_opt, client_models, client_states

    raise ValueError(f'unknown method: {method}')

def vote(labels,classes=11):
    a=np.asarray(labels); out=np.full(a.shape[1],-1,dtype=int)
    for j in range(a.shape[1]):
        v=a[:,j]; v=v[v>=0]; out[j]=np.bincount(v,minlength=classes).argmax() if len(v) else -1
    return out

def aggregate_without_vote(labels,mode='random',rng=None,classes=11):
    """Stand-in for Eq (17) in the "w/o voting" ablation.

    Dropping the majority vote does not remove Algorithm 1 step 5: the clients and
    the server classifier are all distilled on ONE global label sequence P-hat_s, and
    the server classifier is the evaluation model.  So something still has to turn
    the K per-client uploads into a single No-long vector.

    mode='random' (default) -- per-sample "random dictator": among the clients that
        did NOT mark the sample unfamiliar, pick one uniformly at random.  This is
        the only variant that changes Eq (17) and nothing else: the set of samples
        that survive filtering (coverage) is identical to the vote's, and the
        distillation set keeps size No, so the number of gradient steps per round is
        unchanged.  That isolates the majority rule itself, which is what the
        ablation is supposed to measure.
    mode='solo' -- adopt one designated client's upload verbatim.  Cheaper to
        describe, but it silently changes coverage too: samples that client called
        unfamiliar are dropped, so the ablation mixes "no majority" with "no pooling
        of the other clients' familiar samples".

    """
    a=np.asarray(labels)
    if a.ndim!=2:
        raise ValueError(f'expected a (K, No) upload matrix, got shape {a.shape}')
    if a.shape[0]==0: return np.full(a.shape[1],-1,dtype=int)   # nobody uploaded this round
    if mode=='solo':
        k=0 if rng is None else int(rng.integers(a.shape[0]))
        return a[k].astype(int).copy()
    if mode=='random':
        rng=np.random.default_rng() if rng is None else rng
        out=np.full(a.shape[1],-1,dtype=int)
        for j in range(a.shape[1]):
            v=a[:,j]; v=v[v>=0]
            out[j]=-1 if len(v)==0 else int(v[rng.integers(len(v))])
        return out
    raise ValueError(f'unknown no-voting mode: {mode}')

def metrics(y,p):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    return {'accuracy':accuracy_score(y,p),'precision':precision_score(y,p,average='macro',zero_division=0),'recall':recall_score(y,p,average='macro',zero_division=0),'f1':f1_score(y,p,average='macro',zero_division=0)}

def communication(model, clients, n_open, classes=11, method='ssfl'):
    """Per-round bytes under the paper's accounting (Section V-E).

    Only FL communicates parameters; the three distillation methods exchange
    logits / logit summaries / hard labels, so their cost does not scale with the
    model size.  ``model`` may be ``None`` for FD, which has no server model.
    """
    if method=='fl': return 2*sum(p.numel() for p in model.parameters())*4*len(clients)
    if method=='fd': return 2*classes*classes*8*len(clients)  # L per-class vectors, each of dim L
    if method=='dsfl': return 2*n_open*classes*8*len(clients)
    if method=='ssfl': return 2*n_open*len(clients)  # hard labels, one byte each way
    raise ValueError(f'unknown method: {method}')
