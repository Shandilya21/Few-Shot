"""
Implementation of fashionNet results of Snell et al Prototypical networks.

"""
import sys
sys.path.append('../')
from torch.optim import Adam
from torch.utils.data import DataLoader
import argparse
from few_shot.datasets import fashionNet
from few_shot.models import get_few_shot_encoder
from few_shot.core import NShotTaskSampler, EvaluateFewShot, prepare_nshot_task
from few_shot.proto import proto_net_episode
from few_shot.train import fit
from few_shot.callbacks import *
from few_shot.utils import setup_dirs
from config import PATH

setup_dirs()
assert torch.cuda.is_available()
device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

##############
# Parameters #
##############
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='fashionNet')
parser.add_argument('--distance', default='l2')
parser.add_argument('--n-train', default=1, type=int)
parser.add_argument('--n-test', default=1, type=int)
parser.add_argument('--k-train', default=60, type=int)
parser.add_argument('--k-test', default=5, type=int)
parser.add_argument('--q-train', default=5, type=int)
parser.add_argument('--q-test', default=1, type=int)
args = parser.parse_args()

evaluation_episodes = 100
episodes_per_epoch = 10

if args.dataset == 'fashionNet':
	n_epochs = 50
	dataset_class = fashionNet
	num_input_channels = 3
	drop_lr_every = 30
else:
    raise(ValueError, 'Unsupported dataset')

param_str = '{}_nt={}_kt={}_qt={}_'\
            'nv={}_kv={}_qv={}'.format(args.dataset, args.n_train, args.k_train, args.q_train, args.q_train, args.n_test, args.k_test, args.q_test)

print(param_str)

###################
# Create datasets #
###################
background = dataset_class('background')
background_taskloader = DataLoader(
    background,
    batch_sampler=NShotTaskSampler(background, episodes_per_epoch, args.n_train, args.k_train, args.q_train),
    num_workers=4
)
evaluation = dataset_class('evaluation')
evaluation_taskloader = DataLoader(
    evaluation,
    batch_sampler=NShotTaskSampler(evaluation, episodes_per_epoch, args.n_test, args.k_test, args.q_test),
    num_workers=4
)

#########
# Model #
#########
model = get_few_shot_encoder(num_input_channels)
model.to(device, dtype=torch.float)


############
# Training #
############
print('Training Prototypical network on {}...'.format(args.dataset))
optimiser = Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.NLLLoss().cuda()


def lr_schedule(epoch, lr):
    # Drop lr every 2000 episodes
    if epoch % drop_lr_every == 0:
        return lr / 2
    else:
        return lr


callbacks = [
    EvaluateFewShot(
        eval_fn=proto_net_episode,
        num_tasks=evaluation_episodes,
        n_shot=args.n_test,
        k_way=args.k_test,
        q_queries=args.q_test,
        taskloader=evaluation_taskloader,
        prepare_batch=prepare_nshot_task(args.n_test, args.k_test, args.q_test),
        distance=args.distance
    ),
    ModelCheckpoint(
        filepath=PATH + '/models/proto_nets/{}.pth'.format(param_str),
        monitor='val_{}-shot_{}-way_acc'.format(args.n_test, args.k_test)
    ),
    LearningRateScheduler(schedule=lr_schedule),
    CSVLogger(PATH + '/logs/proto_nets/{}.csv'.format(param_str)),
]

fit(
    model,
    optimiser,
    loss_fn,
    epochs=n_epochs,
    dataloader=background_taskloader,
    prepare_batch=prepare_nshot_task(args.n_train, args.k_train, args.q_train),
    callbacks=callbacks,
    metrics=['categorical_accuracy'],
    fit_function=proto_net_episode,
    fit_function_kwargs={'n_shot': args.n_train, 'k_way': args.k_train, 'q_queries': args.q_train, 'train': True,
                         'distance': args.distance},
)
