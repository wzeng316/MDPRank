
set -o errexit


for Fold in Fold1 Fold2 Fold3 Fold4 Fold5
    do
       nohup python train.py $Fold >out_$Fold 2>&1 &
     done

