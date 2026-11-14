"""pi0 entry point for the shared native pi0/pi05 multi-GPU evaluator."""
from evaluation.pi05_multigpu import main


if __name__ == '__main__':
    main(default_config='pi0.example.json')
