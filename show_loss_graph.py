#
# I referred to http://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_training_curves.html
#

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

# different learning rate schedules and momentum parameters
params = [{'solver': 'sgd', 'learning_rate': 'constant', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'constant', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': 0,
           'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': True, 'learning_rate_init': 0.2},
          {'solver': 'sgd', 'learning_rate': 'invscaling', 'momentum': .9,
           'nesterovs_momentum': False, 'learning_rate_init': 0.2},
          {'solver': 'adam', 'learning_rate_init': 0.01}]

labels = ["constant learning-rate",
          "constant with momentum",
          "constant with Nesterov's momentum",
          "inv-scaling learning-rate",
          "inv-scaling with momentum",
          "inv-scaling with Nesterov's momentum",
          "adam"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '-'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '--'},
             {'c': 'black', 'linestyle': '-'}]


def example_plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X)
    mlps = []
    if name == "digits":
        # digits is larger but converges fairly quickly
        max_iter = 15
    else:
        max_iter = 400

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)


def example_main():

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # load / generate some toy datasets
    iris = datasets.load_iris()
    digits = datasets.load_digits()

    data_sets = [(iris.data, iris.target),
                 (digits.data, digits.target),
                 datasets.make_circles(noise=0.2, factor=0.5, random_state=1),
                 datasets.make_moons(noise=0.3, random_state=0)]

    for ax, data, name in zip(axes.ravel(), data_sets, ['iris', 'digits',
                                                        'circles', 'moons']):
        example_plot_on_dataset(*data, ax=ax, name=name)

    fig.legend(ax.get_lines(), labels, ncol=3, loc="upper center")
    plt.show()


def show_loss_graph(X, y):
    # plot_on_dataset(X, y)

    # for each dataset, plot learning for each learning strategy
    plt.title('loss')
    mlps = []

    max_iter = 200

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(verbose=0, random_state=0,
                            max_iter=max_iter, **param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)

    for mlp, label, args in zip(mlps, labels, plot_args):
        plt.plot(mlp.loss_curve_, label=label, **args)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    example_main()