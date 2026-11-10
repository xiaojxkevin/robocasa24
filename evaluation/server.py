"""Run with the selected model project's Python; bind locally by default."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import zmq
from evaluation.protocol import PROTOCOL, validate_actions


def load_policy(config):
    if config.get('action_semantics') != 'robosuite_raw7':
        raise ValueError('A checkpoint trained to emit robosuite_raw7 is required; joint/absolute targets are unsupported')
    if config.get('gripper_sign', 1) not in (-1, 1):
        raise ValueError('gripper_sign must be -1 or 1')
    backend = config['backend']
    if backend != 'xwam':
        if config.get('state_representation') not in ('state8', 'state13'):
            raise ValueError('Declare state_representation: state8 or state13')
        project = Path(config['project']).resolve()
        sys.path.insert(0, str(project))
        sys.path.insert(0, str(project / 'src'))
    if backend in ('pi0', 'pi05', 'spatial_forcing'):
        from evaluation.adapters.openpi import OpenPiPolicy
        return OpenPiPolicy(config)
    if backend == 'xwam':
        from evaluation.adapters.xwam import XWamPolicy
        return XWamPolicy(config)
    if backend == 'fastwam':
        if config.get('image_layout_contract') != 'left_top_right_wrist_bottom_320x384':
            raise ValueError('FastWAM training montage must match left_top_right_wrist_bottom_320x384')
        from evaluation.adapters.fastwam import FastWamPolicy
        return FastWamPolicy(config)
    raise ValueError(f'Unknown backend: {backend}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--bind', default='tcp://127.0.0.1:9100')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    policy = load_policy(config)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(args.bind)
    print(f'Policy ready: {args.bind}', flush=True)
    try:
        while True:
            request = socket.recv_pyobj()
            try:
                method = request['method']
                if method == 'metadata':
                    result = {'protocol': PROTOCOL, 'config': config,
                              'parallel_safe': config['backend'] == 'xwam'}
                elif method == 'reset':
                    policy.reset(request['episode'])
                    result = None
                elif method == 'predict':
                    result = validate_actions(policy.predict(request['observation']))
                else:
                    raise ValueError(f'Unknown method: {method}')
                socket.send_pyobj({'result': result})
            except Exception:
                socket.send_pyobj({'error': traceback.format_exc()})
    finally:
        if hasattr(policy, 'close'):
            policy.close()
        socket.close()
        context.term()


if __name__ == '__main__':
    main()
