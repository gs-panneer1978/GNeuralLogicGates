#pragma once

constexpr int defNeuralNetOCL = 0x7780; 
constexpr int defObjectsList = 0x7792; // Base type for all object lists
constexpr int defConnect = 0x7781;
constexpr int defArrayConnects = 0x7782;
constexpr int defNeuronBase = 0x7783;
constexpr int defNeuron = 0x7784;
constexpr int defNeuronConv = 0x7785;
constexpr int defNeuronProof = 0x7786;
constexpr int defLayer = 0x7787;
constexpr int defArrayLayer = 0x7788;
constexpr int defNet = 0x7789;
constexpr int defNetConv = 0x7790;
constexpr int defNeuronLSTM = 0x7791;
//---
constexpr int defBufferDouble = 0x7882;
constexpr int defNeuronBaseOCL = 0x7883;
constexpr int defNeuronLSTMOCL = 0x7884;
//---
constexpr int def_k_FeedForward = 0;
constexpr int def_k_ff_matrix_w = 0;
constexpr int def_k_ff_matrix_i = 1;
constexpr int def_k_ff_matrix_o = 2;
constexpr int def_k_ff_inputs = 3;
constexpr int def_k_ff_activation = 4;
//---
constexpr int def_k_CaclOutputGradient = 1;
constexpr int def_k_cog_matrix_t = 0;
constexpr int def_k_cog_matrix_o = 1;
constexpr int def_k_cog_matrix_ig = 2;
constexpr int def_k_cog_activation = 3;
//---
constexpr int def_k_CaclHiddenGradient = 2;
constexpr int def_k_chg_matrix_w = 0;
constexpr int def_k_chg_matrix_g = 1;
constexpr int def_k_chg_matrix_o = 2;
constexpr int def_k_chg_matrix_ig = 3;
constexpr int def_k_chg_outputs = 4;
constexpr int def_k_chg_activation = 5;
//---
constexpr int def_k_UpdateWeightsMomentum = 3;
constexpr int def_k_uwm_matrix_w = 0;
constexpr int def_k_uwm_matrix_g = 1;
constexpr int def_k_uwm_matrix_i = 2;
constexpr int def_k_uwm_matrix_dw = 3;
constexpr int def_k_uwm_inputs = 4;
constexpr int def_k_uwm_learning_rates = 5;
constexpr int def_k_uwm_momentum = 6;
//---
constexpr int def_k_UpdateWeightsAdam = 4;
constexpr int def_k_uwa_matrix_w = 0;
constexpr int def_k_uwa_matrix_g = 1;
constexpr int def_k_uwa_matrix_i = 2;
constexpr int def_k_uwa_matrix_m = 3;
constexpr int def_k_uwa_matrix_v = 4;
constexpr int def_k_uwa_inputs = 5;
constexpr int def_k_uwa_l = 6;
constexpr int def_k_uwa_b1 = 7;
constexpr int def_k_uwa_b2 = 8;
//---
constexpr double b1 = 0.99;
constexpr double b2 = 0.9999;