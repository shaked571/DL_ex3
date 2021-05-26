#!/bin/bash
which python

for part in a b c d
do
 for task in ner pos
 do
    for hidden_dim in 200 300
    do
      for optimizer in AdamW
      do
      for l_r in  0.001 0.003 0.005
      do
       for batch_size in 16 32 64 128
       do
       for lhd in 30 50 100
       do

       echo "Output:"
        echo "part${part}_task${task}_hiddendim${hidden_dim}_optim_${optimizer}_lr${l_r}_batch_size${batch_size}"

         python bilstmTrain.py \
         "${part}" \
         "${task}_${part}_hd_${hidden_dim}_b_${batch_size}" \
         "data/${task}/train" \
         -t "${task}" \
         -dev "data/${task}/dev" \
         --o $optimizer \
         -b $batch_size \
         -hd $hidden_dim \
         -l $l_r \
         -lhd $lhd \
          -e 5 -l 0.004 -b 64

      done
     done
    done
   done
  done
 done
 done