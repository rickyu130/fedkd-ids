import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import copy
import numpy as np
import torch
from torch import nn
from core import vote, aggregate_without_vote, partition, partition_fedkd1, CNN, MLP, LSTM, discriminator_labels, client_skip_reason, build_model, federated_round, train, set_seed
from prepare_data import window_feature_permutation
from run import extended_metrics


def test_cnn_matches_the_paper_conv1d_downsampling_schedule():
    """Table I specifies a 23-channel Conv1D stack with two stride-2 layers."""
    model = CNN()
    convs = [module for module in model.modules() if isinstance(module, nn.Conv1d)]
    assert [(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride) for conv in convs] == [
        (23, 64, (3,), (1,)), (64, 64, (3,), (1,)),
        (64, 64, (3,), (1,)), (64, 64, (3,), (1,)),
        (64, 128, (3,), (1,)), (128, 128, (3,), (1,)),
        (128, 128, (3,), (2,)), (128, 128, (3,), (2,)),
    ]
    assert sum(isinstance(module, nn.ReLU) for module in model.features) == 7
    assert sum(isinstance(module, nn.Dropout) for module in model.features) == 3
    model(torch.zeros(2, 23, 5))
    linears = [module for module in model.modules() if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [(256, 128), (128, 11)]


def test_fl_round_weights_models_by_client_sample_count():
    """FedAvg's global model is the sample-count-weighted local-model average."""
    x = np.stack([np.full((23, 5), fill_value, dtype=np.float32) for fill_value in range(4)])
    y = np.array([0, 1, 1, 1], dtype=np.int64)
    clients = [np.array([0]), np.array([1, 2, 3])]
    set_seed(13)
    base = MLP(classes=2)

    set_seed(29)
    local_models = []
    for ids in clients:
        local = copy.deepcopy(base)
        local, _ = train(local, x[ids], y[ids], epochs=1, lr=1e-3, batch=4, device='cpu')
        local_models.append(local)
    expected = {
        name: sum(len(ids) * local.state_dict()[name] for ids, local in zip(clients, local_models)) / len(x)
        for name in base.state_dict()
    }

    set_seed(29)
    actual, _, _, _ = federated_round(copy.deepcopy(base), x, y, clients, 'fl',
                                      local_epochs=1, lr=1e-3, batch=4, device='cpu')
    for name, value in actual.state_dict().items():
        assert torch.allclose(value, expected[name]), name


def test_vote_ignores_unfamiliar():
    assert np.array_equal(vote([[0, -1, 2],[1,1,-1],[1,2,2]]),[1,1,2])
def test_scenario_one_two_shards():
    x=np.zeros((330,23,5)); y=np.repeat(np.arange(11),30); dev=np.repeat(1,330)
    c=partition(x,y,dev,1); assert len(c)==3 and sum(map(len,c))==330

def test_scenario_two_uses_device_class_count():
    x=np.zeros((170,23,5)); y=np.repeat(np.arange(10),17); dev=np.repeat([1,2],85)
    assert len(partition(x,y,dev,2))==10

def test_scenario_three_routes_every_sample():
    x=np.zeros((3000,23,5)); y=np.repeat(np.arange(10),300); dev=np.repeat(1,3000)
    for seed in range(5): assert sum(map(len,partition(x,y,dev,3,seed)))==3000

def test_discriminator_returns_boolean_mask():
    m=CNN(); m(torch.zeros(1,23,5))
    mask,disc,disc_opt=discriminator_labels(m,np.zeros((4,23,5),np.float32),np.zeros((3,23,5),np.float32),.5,epochs=1,batch=4)
    assert mask.shape==(3,) and disc(torch.zeros(1,23,5)).shape[-1]==2 and disc_opt is not None


def test_eval_predictions_are_deterministic_and_restore_training_state():
    from core import predict_proba_eval
    model = CNN(); model(torch.zeros(1,23,5)); model.train()
    x = np.random.rand(12,23,5).astype(np.float32)
    first = predict_proba_eval(model, x, 'cpu')
    second = predict_proba_eval(model, x, 'cpu')
    assert np.array_equal(first, second)
    assert model.training


def test_fedkd_partition_has_27_non_iid_clients_and_full_coverage():
    x = np.zeros((990,23,5), dtype=np.float32)
    y = np.repeat(np.arange(11), 90)
    devices = np.repeat(np.arange(9), 110)
    clients = partition_fedkd1(x, y, devices, seed=7)
    assert len(clients) == 27
    assert all(2 <= len(np.unique(y[ids])) <= 7 for ids in clients)
    flat = np.concatenate(clients)
    assert len(flat) == len(x) and len(np.unique(flat)) == len(x)

def test_ssfl_discriminator_persists_across_rounds():
    """The discriminator is part of the client's persistent state.

    Algorithm 1 step 2 says "train wk,d with Dk,d" -- the same verb used for the
    classifier in step 1 -- and the symbol table states that "each client has a
    discriminator wk,d".  It must therefore be warm-started across rounds rather
    than rebuilt from a fresh random initialisation every round.
    """
    torch.manual_seed(0)
    n = 30
    x = np.random.rand(n, 23, 5).astype(np.float32)
    y = np.random.randint(0, 11, n).astype(np.int64)
    clients = partition(x, y, np.repeat(1, n), 1, 0)
    open_x = np.random.rand(20, 23, 5).astype(np.float32)
    m = CNN(); m(torch.zeros(1, 23, 5))
    _, sopt, client_models, states = federated_round(m, x, y, clients, 'ssfl', open_x, 1, 1e-4, 100, 'cpu')
    assert len(states) == len(clients) and all(s.disc is not None for s in states)
    before = [p.detach().clone() for p in states[0].disc.parameters()]
    _, _, _, states2 = federated_round(m, x, y, clients, 'ssfl', open_x, 1, 1e-4, 100, 'cpu',
                                       client_models=client_models, client_states=states, server_opt=sopt)
    assert states2[0].disc is states[0].disc, 'discriminator must be reused, not rebuilt'
    after = [p.detach().clone() for p in states2[0].disc.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(before, after)), 'must keep training'

def test_ssfl_optimiser_persists_across_rounds():
    """Adam's moments must survive across rounds rather than being rebuilt.

    A fresh Adam takes a full-``lr`` first step (the bias correction makes
    mhat/sqrt(vhat) exactly +-1) and needs several steps to settle back to the
    steady-state step size, so rebuilding it every round wastes a warm-up at the
    start of each round and discards accumulated gradient statistics.
    """
    torch.manual_seed(0)
    n = 30
    x = np.random.rand(n, 23, 5).astype(np.float32)
    y = np.random.randint(0, 11, n).astype(np.int64)
    clients = partition(x, y, np.repeat(1, n), 1, 0)
    open_x = np.random.rand(20, 23, 5).astype(np.float32)
    m = CNN(); m(torch.zeros(1, 23, 5))
    _, sopt, client_models, states = federated_round(m, x, y, clients, 'ssfl', open_x, 1, 1e-4, 100, 'cpu')
    assert states[0].opt is not None and sopt is not None
    _, sopt2, _, states2 = federated_round(m, x, y, clients, 'ssfl', open_x, 1, 1e-4, 100, 'cpu',
                                           client_models=client_models, client_states=states, server_opt=sopt)
    assert states2[0].opt is states[0].opt, 'classifier optimiser must be reused'
    assert sopt2 is sopt, 'server optimiser must be reused'
    opt = states2[0].opt
    assert any('exp_avg' in opt.state[p] and float(opt.state[p]['exp_avg'].abs().sum()) > 0
               for g in opt.param_groups for p in g['params']), 'moments must be non-zero'

def test_ssfl_honours_independent_round_budgets(monkeypatch):
    """Algorithm 1's discriminator (step 2) and distillation (step 5) get their own
    epoch budgets instead of reusing the local supervised budget of step 1."""
    import core
    torch.manual_seed(0)
    n = 30
    x = np.random.rand(n, 23, 5).astype(np.float32)
    y = np.random.randint(0, 11, n).astype(np.int64)
    clients = partition(x, y, np.repeat(1, n), 1, 0)
    open_x = np.random.rand(20, 23, 5).astype(np.float32)
    disc_epochs, train_epochs = [], []
    orig_dl, orig_train = core.discriminator_labels, core.train
    def record_discriminator(*args, **kwargs):
        disc_epochs.append(kwargs.get('epochs'))
        familiar, disc, disc_opt = orig_dl(*args, **kwargs)
        return np.ones(len(args[2]), dtype=bool), disc, disc_opt
    monkeypatch.setattr(core, 'discriminator_labels', record_discriminator)
    monkeypatch.setattr(core, 'train',
                        lambda m, xx, yy, epochs=5, lr=1e-4, batch=100, device='cpu', opt=None:
                            train_epochs.append(epochs) or orig_train(m, xx, yy, epochs, lr, batch, device, opt=opt))
    m = CNN(); m(torch.zeros(1, 23, 5))
    federated_round(m, x, y, clients, 'ssfl', open_x, 2, 1e-4, 100, 'cpu',
                    dist_rounds=9, discri_rounds=7)
    assert disc_epochs and all(e == 7 for e in disc_epochs), 'discri_rounds must drive the discriminator'
    assert 2 in train_epochs and 9 in train_epochs, 'step 1 uses local_epochs, step 5 uses dist_rounds'

def test_build_model_selects_requested_architecture():
    assert isinstance(build_model('cnn'), CNN)
    assert isinstance(build_model('mlp'), MLP)
    assert isinstance(build_model('lstm'), LSTM)

def test_window_feature_permutation_groups_five_windows_into_23_rows():
    windows = ['L5', 'L3', 'L1', 'L0.1', 'L0.01']
    specs = {'MI_dir': ['weight', 'mean', 'variance'], 'H': ['weight', 'mean', 'variance'],
             'HH': ['weight', 'mean', 'std', 'magnitude', 'radius', 'covariance', 'pcc'],
             'HH_jit': ['weight', 'mean', 'variance'],
             'HpHp': ['weight', 'mean', 'std', 'magnitude', 'radius', 'covariance', 'pcc']}
    cols = [f'{block}_{window}_{stat}' for block, stats in specs.items()
            for window in windows for stat in stats]
    perm = window_feature_permutation(cols)
    reordered = [cols[i] for i in perm]
    assert len(perm) == 115
    assert all('_L5_' in c for c in reordered[:23])
    assert [reordered[i * 23] for i in range(5)] == [
        'MI_dir_L5_weight', 'MI_dir_L3_weight', 'MI_dir_L1_weight',
        'MI_dir_L0.1_weight', 'MI_dir_L0.01_weight']


def test_extended_metrics_reports_macro_precision_recall_and_f1():
    metrics, _ = extended_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), classes=2)
    assert metrics['macro_precision'] == 0.8333333333333333
    assert metrics['macro_recall'] == 0.75
    assert metrics['macro_f1'] == 0.7333333333333334

def test_aggregate_without_vote_random_picks_a_contributed_label():
    labs = np.array([[0, -1, 2], [1, 1, -1], [1, 2, 2]])
    out = aggregate_without_vote(labs, mode='random', rng=np.random.default_rng(0))
    # sample 1: only clients 1,2 contributed -> {1,2}; sample 0 -> {0,1}; sample 2 -> {2}
    assert out[2] == 2
    assert out[0] in (0, 1) and out[1] in (1, 2)

def test_aggregate_without_vote_random_keeps_the_vote_coverage():
    labs = np.array([[0, -1, 2], [1, 1, -1], [-1, 2, 2]])
    out = aggregate_without_vote(labs, mode='random', rng=np.random.default_rng(1))
    # coverage (which samples get a label at all) must match the majority vote's,
    # otherwise the ablation changes more than Eq (17).
    assert np.array_equal(out >= 0, vote(labs) >= 0)
    assert (out >= 0).all()

def test_aggregate_without_vote_marks_fully_unfamiliar_samples():
    labs = np.array([[-1, 0], [-1, -1], [-1, 3]])
    assert aggregate_without_vote(labs, rng=np.random.default_rng(2))[0] == -1

def test_aggregate_without_vote_solo_returns_one_client_upload_verbatim():
    labs = np.array([[0, 5, 2], [1, 1, -1], [7, 2, 2]])
    out = aggregate_without_vote(labs, mode='solo', rng=np.random.default_rng(0))
    assert any(np.array_equal(out, row) for row in labs)   # verbatim, vote would mix rows
    assert np.array_equal(out, aggregate_without_vote(labs, mode='solo', rng=np.random.default_rng(0)))
    assert np.array_equal(aggregate_without_vote(labs, mode='solo'), labs[0])  # no rng -> client 0

def test_aggregate_without_vote_is_reproducible_and_seed_sensitive():
    labs = np.array([[0, 4, 2], [1, 1, 3], [3, 2, 2]])
    a1 = aggregate_without_vote(labs, rng=np.random.default_rng([7, 0]))
    a2 = aggregate_without_vote(labs, rng=np.random.default_rng([7, 0]))
    assert np.array_equal(a1, a2)
    seen = {tuple(aggregate_without_vote(labs, rng=np.random.default_rng(s))) for s in range(12)}
    assert len(seen) > 1   # the pick actually varies with the seed

def test_aggregate_without_vote_rejects_unknown_mode():
    try:
        aggregate_without_vote(np.array([[0, 1]]), mode='whatever')
    except ValueError:
        return
    raise AssertionError('expected ValueError')

def test_federated_round_fl_weights_clients_by_private_sample_count(monkeypatch):
    """FL Eq. (1) uses N_k/N, rather than an unweighted client mean."""
    import core
    base = torch.nn.Linear(1, 1, bias=False)
    x = np.zeros((4, 1), dtype=np.float32)
    y = np.zeros(4, dtype=np.int64)
    clients = [np.array([0]), np.array([1, 2, 3])]

    def fake_train(model, xx, yy, *args, **kwargs):
        with torch.no_grad(): model.weight.fill_(float(len(xx)))
        return model, None

    monkeypatch.setattr(core, 'train', fake_train)
    out, _, _, _ = core.federated_round(base, x, y, clients, 'fl', None, 1, 1e-4, 4, 'cpu')
    assert float(out.weight.item()) == 2.5  # (1*1 + 3*3) / 4

def test_client_skip_reason_flags_only_the_two_degenerate_cases():
    assert client_skip_reason(0) == 'no private samples'          # (11) has nothing to train on
    assert client_skip_reason(10, 0) == 'no unfamiliar samples for D_k,d'
    assert client_skip_reason(10, 5) is None
    assert client_skip_reason(10) is None                          # unfamiliar count not computed yet

def test_ssfl_skips_client_that_cannot_build_a_two_class_discriminator():
    """A zero-sample client must sit out: Eq (14) would add no "familiar" half.

    D_k,d would then hold only the unfamiliar class, so the discriminator's arg max
    in Eq (15) is arbitrary and the client's upload would be noise.
    """
    torch.manual_seed(0)
    n = 30
    x = np.random.rand(n, 23, 5).astype(np.float32)
    y = np.random.randint(0, 11, n).astype(np.int64)
    clients = partition(x, y, np.repeat(1, n), 1, 0)
    clients.append(np.zeros(0, dtype=int))                         # a client the Dirichlet split can produce
    open_x = np.random.rand(20, 23, 5).astype(np.float32)
    m = CNN(); m(torch.zeros(1, 23, 5))
    stats = {}
    server, _, client_models, _ = federated_round(m, x, y, clients, 'ssfl', open_x, 1, 1e-4, 100, 'cpu', stats=stats)
    assert (len(clients) - 1, 'no private samples') in stats['skipped']
    assert all(torch.isfinite(p).all() for p in server.parameters()), 'server must not go NaN'
    assert all(torch.isfinite(p).all() for p in client_models[-1].parameters()), 'skipped client must not go NaN'

def test_ssfl_round_with_no_uploads_at_all_is_a_no_op():
    """If every client is skipped the round must still complete.

    A single open sample makes ``conf < median(conf)`` empty for everyone, so no
    client can build a two-class D_k,d.  Nobody votes, every sample stays unknown,
    and the classifiers simply carry over.
    """
    torch.manual_seed(0)
    n = 30
    x = np.random.rand(n, 23, 5).astype(np.float32)
    y = np.random.randint(0, 11, n).astype(np.int64)
    clients = partition(x, y, np.repeat(1, n), 1, 0)
    open_x = np.random.rand(1, 23, 5).astype(np.float32)          # median == the only score
    m = CNN(); m(torch.zeros(1, 23, 5))
    stats = {}
    server, _, _, _ = federated_round(m, x, y, clients, 'ssfl', open_x, 1, 1e-4, 100, 'cpu', stats=stats)
    assert len(stats['skipped']) == len(clients)
    assert all(r == 'no unfamiliar samples for D_k,d' for _, r in stats['skipped'])
    assert all(torch.isfinite(p).all() for p in server.parameters())
