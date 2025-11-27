#!/bin/bash
# Run PKU experiment sweep in tmux with batch mode

bash src/helpers/tmux_wrapper.sh \
    exp12_pku \
    "bash experiments/12_UT_PW-Q_Rec_Pr/bash/2_pku_sweep_other_models.sh"
