import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim, autograd
import data_preprocess as data_pre
import random

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('--hidden_dim', type=int, default=784)
parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--penalty_anneal_iters', type=int, default=100)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--grayscale_model', action='store_true')
flags = parser.parse_args()
models_performance = [[], [], [], [], [], []]


def setup_seed(seed):
    # 设置随机数种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def model_to_int(model_name):
    if model_name[:2] == 'HG':
        return 1
    elif model_name[:2] == 'ST':
        return 2
    elif model_name[:2] == 'WD':
        return 1
    elif model_name[:2] == 'TO':
        return 1
    elif model_name[:2] == 'Sa':
        return 1
    elif model_name[:2] == 'Hi':
        return 1
    else:
        return 0


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.days, self.features = 28, 28
        if flags.grayscale_model:
            lin1 = nn.Linear(28 * 28, flags.hidden_dim)
        else:
            # lin1 = nn.Linear(2 * 28 * 28, flags.hidden_dim)
            lin1 = nn.Linear(28 * 28, flags.hidden_dim)
        lin2 = nn.Linear(flags.hidden_dim, flags.hidden_dim)
        lin3 = nn.Linear(flags.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def forward(self, input):
        if flags.grayscale_model:
            out = input.view(input.shape[0], 2, 28 * 28).sum(dim=1)
        else:
            # out = input.view(input.shape[0], 2 * 28 * 28)
            out = input.view(input.shape[0], 28 * 28).float()
        out = self._main(out)
        return out


def mean_nll(logits, y):
    y = y.view(y.shape[0], -1).float()
    return nn.functional.binary_cross_entropy_with_logits(logits, y)
    # reference https://blog.csdn.net/u010630669/article/details/105599067


def mean_accuracy(logits, y, print_flag=False):
    preds = (logits > 0.).int().view(logits.shape[0])
    # TP    predict 和 label 同时为1
    TP = ((preds.data == 1) & (y.data == 1)).to(device).sum()
    # TN    predict 和 label 同时为0
    TN = ((preds.data == 0) & (y.data == 0)).to(device).sum()
    # FN    predict 0 label 1
    FN = ((preds.data == 0) & (y.data == 1)).to(device).sum()
    # FP    predict 1 label 0
    FP = ((preds.data == 1) & (y.data == 0)).to(device).sum()

    p = TP / (TP + FP + 1e-15)
    r = TP / (TP + FN + 1e-15)
    FDR = TP / (TP + FN + 1e-15)
    FAR = FP / (FP + TN + 1e-15)
    F1 = 2 * r * p / (r + p + 1e-15)
    acc = (TP + TN) / (TP + TN + FP + FN + 1e-15)
    if print_flag:
        print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
        print('FDR=', FDR.data, 'FAR=', FAR.data, 'F1=', F1.data, 'acc=', acc.data)
        return acc.data, FDR.data, FAR.data
    elif not print_flag:
        return acc.data


def penalty(logits, y):
    scale = torch.tensor(1.).to(device).requires_grad_()
    loss = mean_nll(logits * scale, y)
    grad = autograd.grad(loss, [scale], create_graph=True)[
        0]  # reference https://blog.csdn.net/qq_36556893/article/details/91982925
    return torch.sum(grad ** 2)


def pretty_print(*values):
    col_width = 13

    def format_val(v):
        if not isinstance(v, str):
            v = np.array2string(v, precision=5, floatmode='fixed')
        return v.ljust(col_width)

    str_values = [format_val(v) for v in values]
    print("   ".join(str_values))


def wjc_make_environment(images_data, targets_data, name_data):
    # Build environments
    mid_envs = []  # return envs
    for j in range(len(env_names)):
        mid_feature = env_names[j]  # mid_feature: current feature filter.
        mid_env_images, mid_env_labels = None, None
        for name_i in range(len(name_data)):
            if k_labels[name_i] == mid_feature:
                if mid_env_images is None:
                    mid_env_images = images_data[name_i].clone().detach()
                    mid_env_images = mid_env_images.unsqueeze(0)
                else:
                    mid_env_images = torch.cat([mid_env_images, images_data[name_i].unsqueeze(0).clone().detach()], 0)
                if mid_env_labels is None:
                    mid_env_labels = torch.tensor([targets_data[name_i]])
                elif mid_env_labels is not None:
                    mid_env_labels = torch.cat([mid_env_labels, torch.tensor([targets_data[name_i]])], 0)

        # print('env[' + str(j) + '] is ', mid_env_images.shape, sum(mid_env_labels), 'broken')
        mid_envs.append({"images": mid_env_images, "labels": mid_env_labels})
    return mid_envs


k = 5  # k = data_pre.test_k_Kmeans_env_tag(wjc_data)
env_names, k_labels = data_pre.Kmeans_env_tag(torch.load('./real_dataset/wjc_data_1.pt') / 255,
                                              torch.load('./real_dataset/this_dataset_1.pt'), k, True)

# envs names
print('different environment feature values:', k)

for restart in range(flags.n_restarts):
    print("Restart", restart)
    setup_seed(restart)
    final_train_accs = []
    final_test_accs = []

    wjc_data, wjc_target, this_dataset = torch.load('./real_dataset/wjc_data_1.pt') / 255, torch.load(
        './real_dataset/wjc_target_1.pt'), torch.load('./real_dataset/this_dataset_1.pt')
    this_dataset = list(this_dataset)

    mid_seed = random.random()
    random.seed(mid_seed)
    random.shuffle(wjc_data)
    random.seed(mid_seed)
    random.shuffle(wjc_target)
    random.seed(mid_seed)
    random.shuffle(this_dataset)
    random.seed(mid_seed)
    random.shuffle(k_labels)
    random.seed(None)

    # ERM model environment builder
    # envs = wjc_make_environment(wjc_data, wjc_target, this_dataset)
    # envs = [envs[0], envs[1], envs[2], envs[4], envs[3]]  # set env[2] as test set
    # test_set = envs[-1].copy()

    # ERM model environment builder
    envs = wjc_make_environment(wjc_data, wjc_target, this_dataset)
    envs = [envs[4], envs[1], envs[2], envs[3], envs[0]]  # set env[2] as test set

    mid_images, mid_labels = None, None
    for i in range(len(envs) - 1):
        env = envs[i]
        # print(env['images'].shape, env['labels'].shape)
        if mid_images is None:
            mid_images = env['images'].clone()
        elif mid_images is not None:
            mid_images = torch.cat([mid_images, env['images']], 0)

        if mid_labels is None:
            mid_labels = env['labels'].clone()
        elif mid_labels is not None:
            mid_labels = torch.cat([mid_labels, env['labels']], 0)
    envs = [{"images": mid_images, "labels": mid_labels}, envs[-1]]
    test_set = envs[-1].copy()

    # # Another ERM model environment builder
    # train_set = {"images": mnist_train[0], "labels": mnist_train[1]}
    # test_set = {"images": mnist_val[0], "labels": mnist_val[1]}
    # envs = [train_set, test_set]

    mlp = MLP().to(device)
    optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    for step in range(flags.steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

        mid_train_nll, mid_train_acc, mid_train_penalty = [], [], []
        for i_env in range(len(envs) - 1):
            mid_train_nll.append(envs[i_env]['nll'])
            mid_train_acc.append(envs[i_env]['acc'])
            mid_train_penalty.append(envs[i_env]['penalty'])

        train_nll = torch.stack(mid_train_nll).mean()
        train_acc = torch.stack(mid_train_acc).mean()
        train_penalty = torch.stack(mid_train_penalty).mean()

        weight_norm = torch.tensor(0.).to(device)
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += flags.l2_regularizer_weight * weight_norm

        optimizer.zero_grad()  # reference https://blog.csdn.net/scut_salmon/article/details/82414730
        loss.backward()
        optimizer.step()

        test_acc = envs[-1]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy(),
            )
        if step >= flags.penalty_anneal_iters:
            final_train_accs.append(train_acc.detach().cpu().numpy())
            final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))
    models_performance[0].append(np.mean(final_train_accs))
    models_performance[1].append(np.std(final_train_accs))
    models_performance[3].append(np.std(final_test_accs))

    # torch.save(mlp, './models/irm_mlp_seed' + str(restart) + '.pkl')
    logits = mlp(test_set['images'])
    print('Final test set performance (acc) (Vanilla MLP):')
    test_set['acc'] = mean_accuracy(logits, test_set['labels'])
    models_performance[2].append(test_set['acc'])

print('\nModels train acc (mean/std across restarts):')
print(np.mean(models_performance[0]), np.mean(models_performance[1]))
print('Models test acc (mean/std across restarts):')
print(np.mean(models_performance[2]), np.mean(models_performance[3]))

plt.plot(np.arange(1, flags.n_restarts + 1), models_performance[0], 'b', label='Train')
plt.plot(np.arange(1, flags.n_restarts + 1), models_performance[2], 'r', label='Test')
plt.title('Vanilla MLP model')
plt.xlabel('restart')
plt.ylabel('accuracy')
plt.ylim((0, 1.1))
plt.legend()
plt.show()

end = time.time()
print("totally cost ", end - start, " seconds")
