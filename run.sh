#!/bin/bash
which python

for part in a b c d
do
 for task in ner pos
 do
    for hidden_dim in 100 200
    do
      for optimizer in AdamW
      do
      for l_r in  0.001  0.003
      do
       for batch_size in 4 8 16
       do

       echo "Output:"
        echo "part${part}_task${task}_hiddendim${hidden_dim}_optim_${optimizer}_lr${l_r}_batch_size${batch_size}"

         python bilstmTrain.py \
         "${part}" \
         "data/${task}/train" \
         "${task}_${part}_hd_${hidden_dim}_b_${batch_size}_l_r_${l_r}" \
         -t "${task}" \
         -dev "data/${task}/dev" \
         --o $optimizer \
         -b $batch_size \
         -hd $hidden_dim \
         -l $l_r \
         -lhd 200 \
          -e 5

     done
    done
   done
  done
 done
 done