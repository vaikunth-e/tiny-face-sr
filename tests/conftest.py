"""Keep synthetic neural checks small and deterministic on CPU"""


def pytest_sessionstart(session):
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass
