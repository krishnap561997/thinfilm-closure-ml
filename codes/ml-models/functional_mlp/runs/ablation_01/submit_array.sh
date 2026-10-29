#!/bin/bash
# Submit one network size at a time.
echo 'sbatch /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/functional_mlp/runs/ablation_01/layers4_width64/submit_array.sh'
echo 'sbatch /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/functional_mlp/runs/ablation_01/layers2_width64/submit_array.sh'
echo 'sbatch /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/functional_mlp/runs/ablation_01/layers2_width16/submit_array.sh'
echo 'sbatch /blue/bala1s/krishnap.kalivel/TLFHydrodynamics/thinfilm-closure-ml/codes/ml-models/functional_mlp/runs/ablation_01/layers1_width16/submit_array.sh'
exit 1
