import os
import sys
import argparse
import importlib


prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.tracker.stark_st import STARK_ST_ToolKitEval
from got10k.experiments import ExperimentGOT10k



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
    parser.add_argument('tracker_param', type=str, help='Name of parameter file.')

    args = parser.parse_args()

    param_module = importlib.import_module('lib.test.parameter.{}'.format(args.tracker_name))
    params = param_module.parameters(args.tracker_param)

    # setup tracker
    if args.tracker_name == 'stark_st':
        tracker = STARK_ST_ToolKitEval(params, 'video')

    # run experiments on GOT-10k (validation subset)
    experiment = ExperimentGOT10k('data/got10k', subset='test')
    experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name])