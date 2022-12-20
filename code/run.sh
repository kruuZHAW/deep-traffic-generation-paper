# Script used for the training of the paper

# cd deep_traffic_generation

#FCVAE
# Version 0: 
# python fcvae.py --data_path data/traffic_noga_tilFAF_train.pkl --h_dims 216 128 64 --encoding_dim 64 --lrstep 200 --lr 0.001 --lrgamma 0.5 --gradient_clip_val 0.5 --batch_size 500 --features track groundspeed altitude timedelta --info_features latitude longitude --info_index -1

#TCVAE
# Version 0 : standard
# python tcvae.py --data_path data/traffic_noga_tilFAF_train.pkl --prior standard --encoding_dim 64 --h_dims 64 64 64 --lrstep 200 --lr 0.001 --lrgamma 0.5 --gradient_clip_val 0.5 --batch_size 500 --n_components 1000 --features track groundspeed altitude timedelta --info_features latitude longitude --info_index -1

# Version 1 : vampprior
# python tcvae.py --data_path data/traffic_noga_tilFAF_train.pkl --prior vampprior --encoding_dim 64 --h_dims 64 64 64 --lrstep 200 --lr 0.001 --lrgamma 0.5 --gradient_clip_val 0.5 --batch_size 500 --n_components 1000 --features track groundspeed altitude timedelta --info_features latitude longitude --info_index -1

# cd ..

# python3 generation.py traffic_noga_tilFAF.pkl version_0 version_1
python3 plot.py

