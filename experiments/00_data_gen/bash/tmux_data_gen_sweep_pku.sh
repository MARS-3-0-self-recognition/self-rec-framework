#!/bin/bash
# Run PKU data generation sweep in tmux with batch mode

bash src/helpers/tmux_wrapper.sh \
    data_gen_pku \
    "bash experiments/00_data_gen/bash/data_gen_sweep_pku.sh"
