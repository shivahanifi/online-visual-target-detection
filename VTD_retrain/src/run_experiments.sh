#! /bin/sh
python train_on_objectattention.py --chunk_size 2 --batch_size 4 > logs_train_cs2_bs4.txt
python eval_on_objectattention.py > logs_test_cs2_bs4.txt
python train_on_objectattention.py --chunk_size 2 --batch_size 8 > logs_train_cs2_bs8.txt
python eval_on_objectattention.py > logs_test_cs2_bs8.txt
python train_on_objectattention.py --chunk_size 2 --batch_size 16 > logs_train_cs2_bs16.txt
python eval_on_objectattention.py > logs_test_cs2_bs16.txt
python train_on_objectattention.py --chunk_size 3 --batch_size 4 > logs_train_cs3_bs4.txt
python eval_on_objectattention.py > logs_test_cs3_bs4.txt
python train_on_objectattention.py --chunk_size 3 --batch_size 8 > logs_train_cs3_bs8.txt
python eval_on_objectattention.py > logs_test_cs3_bs8.txt
python train_on_objectattention.py --chunk_size 3 --batch_size 16 > logs_train_cs3_bs16.txt
python eval_on_objectattention.py > logs_test_cs3_bs16.txt
python train_on_objectattention.py --chunk_size 4 --batch_size 4 > logs_train_cs4_bs4.txt
python eval_on_objectattention.py > logs_test_cs4_bs4.txt
python train_on_objectattention.py --chunk_size 4 --batch_size 8 > logs_train_cs4_bs8.txt
python eval_on_objectattention.py > logs_test_cs4_bs8.txt
python train_on_objectattention.py --chunk_size 4 --batch_size 16 > logs_train_cs4_bs16.txt
python eval_on_objectattention.py > logs_test_cs4_bs16.txt
