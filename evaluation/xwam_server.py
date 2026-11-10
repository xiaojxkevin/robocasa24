"""Production entry for the upstream X-WAM server with CPU checkpoint staging."""
import argparse
import importlib.util
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    project = args.project.resolve()
    sys.path.insert(0, str(project))
    os.chdir(project)
    spec = importlib.util.spec_from_file_location('robocasa24_xwam_server', project / 'evaluation/policy_server.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import logging
    import torch
    config = module.tyro.cli(module.Args, args=remaining)
    checkpoint = (Path(config.exp_path) / f'checkpoints/{config.steps}.ckpt/checkpoint/mp_rank_00_model_states.pt').resolve()
    original_load = torch.load
    original_load_state = module.XWAMRunner.load_state_dict
    def load_checkpoint(path, *positional, **kwargs):
        if isinstance(path, (str, Path)) and Path(path).resolve() == checkpoint:
            kwargs['map_location'] = 'cpu'
            kwargs['mmap'] = True
            logging.info('Loading SFT checkpoint through CPU mmap; no duplicate GPU checkpoint')
        return original_load(path, *positional, **kwargs)
    def load_state(self, *positional, **kwargs):
        result = original_load_state(self, *positional, **kwargs)
        torch.cuda.empty_cache()
        logging.info('Model loaded: allocated %.2f GiB, reserved %.2f GiB',
                     torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30)
        return result
    module.XWAMRunner.load_state_dict = load_state
    torch.load = load_checkpoint
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        module.main(config)
    finally:
        torch.load = original_load
        module.XWAMRunner.load_state_dict = original_load_state


if __name__ == '__main__':
    main()
