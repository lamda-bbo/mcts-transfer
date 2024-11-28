import os
import json

designers = [
    'Random',
    'ShuffledGridSearch',
    'RegularizedEvolution',
    'HillClimbing',
    'EagleStrategy',
    'CMAES',
    'BotorchBO',
]


def get_cmd(problem, designer, search_space_id, dataset_id, out_name, length, seed):
    prefix = 'taskset -c {}-{}'.format(args.cpu_start, args.cpu_end)
    cmd = 'python data_gen/data_gen_main.py \
        --problem={} \
        --designer={} \
        --search_space_id={} \
        --dataset_id={} \
        --out_name={} \
        --length={} \
        --seed={}'.format(problem, designer, search_space_id, dataset_id, out_name, length, seed)
    cmd = ' '.join([prefix, cmd])
    return cmd


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True, choices=['hpob', 'synthetic', 'metabo_synthetic', 'real_world_problem'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--smoke_test', action='store_true')
    parser.add_argument('--cpu_start', type=int, required=True)
    parser.add_argument('--cpu_end', type=int, required=True)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'validation'])
    args = parser.parse_args()

    # for synthetic
    if args.mode == 'train':
        s, e = 0, 1
    elif args.mode == 'validation':
        s, e = 50, 60
    else: # test
        s, e = 60, 70

    if args.problem == 'hpob':
        with open('others/hpob-summary-stats/{}-summary-stats.json'.format(args.mode), 'rb') as f:
            summary_stats = json.load(f)
    elif args.problem == 'synthetic':
        bbob_func_names = (
            'NegativeSphere',
        )

        summary_stats = {name: [str(i) for i in range(s, e)] for name in bbob_func_names}
    elif args.problem == 'metabo_synthetic':
        names = ('Branin2', 'Hartmann3')
        summary_stats = {name: [str(i) for i in range(s, e)] for name in names}
    elif args.problem == 'real_world_problem':
        names = (
            'LunarLander',
            'PDE',
            'Optics',
            'RobotPush',
            'Rover',
            'Furuta',
        )
        summary_stats = {name: [str(i) for i in range(s, e)] for name in names}
    else:
        raise NotImplementedError

    # key = list(summary_stats.keys())[0]
    # value = summary_stats[key][: 2]
    # summary_stats = dict()
    # summary_stats[key] = value
    # print(summary_stats)

    if args.smoke_test:
        search_space_id = list(summary_stats.keys())[0]
        dataset_id = summary_stats[search_space_id][0]
        print(search_space_id, dataset_id)
        for designer in designers:
            print(designer)
            os.system(get_cmd(
                args.problem,
                designer, 
                search_space_id, 
                dataset_id, 
                'data/generated_data/smoke_test_{}_{}_{}.json'.format(search_space_id, dataset_id, designer), 
                100,
                seed=args.seed,
            ))
    else:
        seed = args.seed
        failed_cmds = []

        if args.mode == 'train':
            dir_path = 'data/generated_data/{}/seed{}'.format(args.problem, seed)
        else:
            dir_path = 'data/generated_data/{}_{}/seed{}'.format(args.problem, args.mode, seed)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        all_out_names = []
        for designer in designers:
            # print(designer, seed)
            for search_space_id in summary_stats:
                for dataset_id in summary_stats[search_space_id]:
                    print(designer, search_space_id, dataset_id, seed)
                    out_name = dir_path + '/{}_{}_{}_{}.json'.format(designer, search_space_id, dataset_id, seed)
                    cmd = get_cmd(
                        args.problem,
                        designer,
                        search_space_id, 
                        dataset_id,
                        out_name,
                        300,
                        seed=seed,
                    )
                    ret = os.system(cmd)
                    if ret != 0:
                        failed_cmds.append(cmd)

                        print('----------------------------')
                        print('Error: {}'.format(cmd))
                        print('----------------------------')

                    all_out_names.append(out_name)

        print('----------------------------')
        print(failed_cmds)
        print('----------------------------')
