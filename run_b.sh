#!/bin/bash
which python

for part in b
do
 for task in ner pos
 do
    for hidden_dim in 200 300
    do
      for optimizer in AdamW
      do
       for batch_size in 25 50 100
       do
       for lhd in 50 100
       do
       echo "Output:"
        echo "part${part}_task${task}_hiddendim${hidden_dim}_optim_${optimizer}_batch_size${batch_size}"

         python bilstmTrain.py \
         "${part}" \
         "data/${task}/train" \
         "${task}_${part}_hd_${hidden_dim}_lhd_${lhd}_b_${batch_size}" \
         -t "${task}" \
         -dev "data/${task}/dev" \
         --o $optimizer \
         -b $batch_size \
         -hd $hidden_dim \
         -lhd $lhd \
          -e 5
      done
     done
    done
   done
  done
 done
 done