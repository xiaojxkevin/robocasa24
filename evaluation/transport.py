"""Trusted-local-network RPC, isolated from model dependencies."""
import zmq
from evaluation.protocol import PROTOCOL, validate_actions


class RemotePolicy:
    def __init__(self, endpoint, timeout_ms=120000):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self.socket.connect(endpoint)
        try:
            self.metadata = self.call('metadata')
            if self.metadata.get('protocol') != PROTOCOL:
                raise ValueError('Policy protocol mismatch')
        except Exception:
            self.close()
            raise

    def call(self, method, **kwargs):
        self.socket.send_pyobj({'method': method, **kwargs})
        reply = self.socket.recv_pyobj()
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        return reply['result']

    def reset(self, episode):
        self.call('reset', episode=episode)

    def predict(self, observation):
        return validate_actions(self.call('predict', observation=observation))

    def close(self):
        self.socket.close()
        self.context.term()
