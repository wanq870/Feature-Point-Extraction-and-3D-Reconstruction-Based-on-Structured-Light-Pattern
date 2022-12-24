python3 run.py new_dataset output \
--num_epoch 2000 \
--batch_size 16 \
--lr 1e-3 \
--weight_decay 1e-3 \

python3 run.py new_dataset output \
--ckpt output/model.ckpt \
--do_predict \