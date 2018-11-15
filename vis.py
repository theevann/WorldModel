#! /usr/bin/env python

import visdom
import copy
import json
import os
import argparse


def create_log(current_env, new_env=None):
    new_env = current_env if new_env is None else new_env
    vis = visdom.Visdom(env=current_env)
    data = json.loads(vis.get_window_data())
    if len(data) == 0:
        print("NOTHING HAS BEEN SAVED: NOTHING IN THIS ENV - DOES IT EXIST ?")
        return

    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = open(dir_path + '/log/' + new_env + '.log', 'w+')
    bootstrap = {"data": [{"content": {"src": "", "caption": None}, "type": ""}], "win": "window_", "eid": "", "opts": {"width": 0, "height": 0}}

    for datapoint in data.values():
        output = copy.deepcopy(bootstrap)
        output['data'][0]['content'] = datapoint['content']
        output['data'][0]['type'] = datapoint['type']
        output['opts']['height'] = datapoint['height']
        output['opts']['width'] = datapoint['width']
        output['win'] = datapoint['id']
        output['eid'] = new_env

        to_write = json.dumps(["events", output])
        file.write(to_write + '\n')
    file.close()


def load_log(env):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    vis = visdom.Visdom(env=env)
    vis.replay_log(dir_path + '/log/' + env + '.log')


def load_all_log():
    dir_path = os.path.dirname(os.path.realpath(__file__)) + '/log/'
    logs = os.listdir(dir_path)
    vis = visdom.Visdom()
    for log in logs:
        vis.replay_log(dir_path + log)


def load_log_at(path):
    file = os.path.basename(path)
    env = os.path.splitext(file)[0]
    vis = visdom.Visdom(env=env)
    vis.replay_log(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('Load and Save from Visdom'))
    parser.add_argument('-s', '--save', type=str, help='env_name', default='')
    parser.add_argument('-l', '--load', type=str, help='env_name', default='')
    parser.add_argument('-lf', '--load-file', type=str, help='path_to_log', default='')
    args = parser.parse_args()

    if args.save is not '':
        create_log(args.save)

    if args.load is not '':
        if args.load == 'all':
            load_all_log()
        else:
            load_log(args.load)

    if args.load_file is not '':
        load_log_at(args.load_file)
