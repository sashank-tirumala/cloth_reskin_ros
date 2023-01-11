#!/usr/bin/python
import sys
import argparse
import torch
import traceback

def save_model_txt(model, path):
    fout = open(path, 'w')
    for k, v in model['model_sd'].items(): # for image classifier
    # for k, v in model.state_dict().items():
    # for k, v in model.items():
        fout.write(str(k) + '\n')
        # if not isinstance(v, torch.Tensor):
            # fout.write(str(v) + '\n')
        # else:
        fout.write(str(v.tolist()) + '\n')
    fout.close()

def load_model_txt(model, path):
    print('Loading...')
    data_dict = {}
    fin = open(path, 'r')
    i = 0
    odd = 1
    prev_key = None
    while True:
        s = fin.readline().strip()
        if not s:
            break
        if odd:
            prev_key = s
        else:
            # print('Iter', i)
            try:
                val = eval(s)
            except Exception as e:
                print(traceback.format_stack(e))
                data_dict[prev_key] = s
                i += 1
                import IPython; IPython.embed()
                continue
            if type(val) != type([]):
                data_dict[prev_key] = torch.FloatTensor([eval(s)])[0]
            else:
                data_dict[prev_key] = torch.FloatTensor(eval(s))
            i += 1
        odd = (odd + 1) % 2

    # Replace existing values with loaded
    own_state = model.state_dict()
    print('Items:', len(own_state.items()))
    for k, v in data_dict.items():
        if not k in own_state:
            print('Parameter', k, 'not found in own_state!!!')
        else:
            try:
                own_state[k].copy_(v)
            except:
                print('Key:', k)
                print('Old:', own_state[k])
                print('New:', v)
                sys.exit(0)
    print('Model loaded')

if __name__ == '__main__':
    # Run this script with python3 conda 

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path')
    args = parser.parse_args()
    weight_path = args.weight_path

    # model = torch.load(weight_path, map_location=torch.device('cpu'))
    model = torch.load(weight_path, map_location=torch.device('cpu'))
    if '.ckpt' in weight_path:
        model = model['state_dict']
        save_name = weight_path.replace('.ckpt', '.txt')
    elif '.pth' in weight_path:
        save_name = weight_path.replace('.pth', '.txt')
    else:
        save_name = weight_path.replace('.pt', '.txt')
    save_model_txt(model, save_name)
