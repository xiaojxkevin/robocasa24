"""Spatial Forcing entry point for the shared native RoboCasa multi-GPU evaluator."""
from evaluation.client import ROOT
from evaluation.pi05_multigpu import main


if __name__ == '__main__':
    main(default_config='spatial_forcing.example.json',
         default_project=ROOT.parent / 'spatial-forcing/policy/Spatial_Forcing/open_sf')
