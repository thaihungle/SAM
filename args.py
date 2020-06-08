import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model_name', default='sam',
                        help='the name of the model')
    parser.add_argument('-task_json', type=str, default='tasks/nar.json',
                        help='path to json file with task specific parameters')
    parser.add_argument('-log_dir', default='logs/',
                        help='path to log metrics')
    parser.add_argument('-save_dir', default='saved_models/',
                        help='path to file with final model parameters')
    parser.add_argument('-batch_size', type=int, default=128,
                        help='batch size of input sequence during training')
    parser.add_argument('-clip_grad', type=int, default=10,
                        help='clip gradient')
    parser.add_argument('-num_iters', type=int, default=50000,
                        help='number of iterations for training')
    parser.add_argument('-freq_val', type=int, default=200,
                        help='validation frequence')
    parser.add_argument('-num_eval', type=int, default=10,
                        help='number of evaluation')
    parser.add_argument('-mode', type=str, default="train",
                        help='train or test')
    parser.add_argument('-resume', type=str, default=None,
                        help='resume path')

    # todo: only rmsprop optimizer supported yet, support adam too
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for rmsprop optimizer')
    parser.add_argument('-momentum', type=float, default=0.9,
                        help='momentum for rmsprop optimizer')
    parser.add_argument('-alpha', type=float, default=0.95,
                        help='alpha for rmsprop optimizer')
    parser.add_argument('-beta1', type=float, default=0.9,
                        help='beta1 constant for adam optimizer')
    parser.add_argument('-beta2', type=float, default=0.999,
                        help='beta2 constant for adam optimizer')
    return parser
