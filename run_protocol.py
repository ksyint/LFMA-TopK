#!/usr/bin/env python3
"""Train a named paper experiment setting and evaluate each selected checkpoint."""
from experiments.run_grid import argument_parser, main


if __name__ == '__main__':
    main(argument_parser(evaluate=True).parse_args())
