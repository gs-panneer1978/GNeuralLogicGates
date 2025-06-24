#pragma once


#define defConnect         0x7781
#define defArrayConnects   0x7782
#define defNeuronBase      0x7783
#define defNeuron          0x7784
#define defNeuronConv      0x7785
#define defNeuronProof     0x7786
#define defLayer           0x7787
#define defArrayLayer      0x7788
#define defNet             0x7789
#define defNetConv         0x7790
#define defNeuronLSTM      0x7791
//---
#define defBufferDouble    0x7882
#define defNeuronBaseOCL   0x7883
#define defNeuronLSTMOCL   0x7884
//---
#define def_k_FeedForward     0
#define def_k_ff_matrix_w     0
#define def_k_ff_matrix_i     1
#define def_k_ff_matrix_o     2
#define def_k_ff_inputs       3
#define def_k_ff_activation   4
//---
#define def_k_CaclOutputGradient 1
#define def_k_cog_matrix_t       0
#define def_k_cog_matrix_o       1
#define def_k_cog_matrix_ig      2
#define def_k_cog_activation     3
//---
#define def_k_CaclHiddenGradient 2
#define def_k_chg_matrix_w       0
#define def_k_chg_matrix_g       1
#define def_k_chg_matrix_o       2
#define def_k_chg_matrix_ig      3
#define def_k_chg_outputs        4
#define def_k_chg_activation     5
//---
#define def_k_UpdateWeightsMomentum      3
#define def_k_uwm_matrix_w        0
#define def_k_uwm_matrix_g        1
#define def_k_uwm_matrix_i        2
#define def_k_uwm_matrix_dw       3
#define def_k_uwm_inputs          4
#define def_k_uwm_learning_rates  5
#define def_k_uwm_momentum        6
//---
#define def_k_UpdateWeightsAdam   4
#define def_k_uwa_matrix_w        0
#define def_k_uwa_matrix_g        1
#define def_k_uwa_matrix_i        2
#define def_k_uwa_matrix_m        3
#define def_k_uwa_matrix_v        4
#define def_k_uwa_inputs          5
#define def_k_uwa_l               6
#define def_k_uwa_b1              7
#define def_k_uwa_b2              8
//---
#define b1                        0.99
#define b2                        0.9999