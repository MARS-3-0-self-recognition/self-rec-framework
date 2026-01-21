#!/bin/bash
# Run PKU experiment sweep in tmux with batch mode

bash src/helpers/tmux_wrapper.sh \
    exp01_pku \
    "bash experiments/01_AT_PW-C_Rec_Pr/bash/2_pku_sweep_other_models.sh --batch"
