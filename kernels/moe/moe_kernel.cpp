// ============================================================================
// MoE FPGA 内核：DeepSeek V3.2 路由专家（v1：N_EXPERT_SLOTS=1）
//
// 重构：将片外内存访问与计算分离。所有持有 async_mmap 端口的任务
// 只负责发出读写请求，并通过流转发数据；
// 计算任务不持有任何 async_mmap 端口。
// ============================================================================

#include "moe_kernel.h"

static inline float moe_sigmoidf(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}
static inline float moe_siluf(float x) {
    return x * moe_sigmoidf(x);
}

// ============================================================================
// 读取任务：x_gate_reader
//   将 x_mem 中的 N * HIDDEN_VECS 个 INT16 向量流式传给 gate_compute。
//   不执行计算。
// ============================================================================
void x_gate_reader(
    const int N,
    tapa::async_mmap<int16_vec_t>& x_mem,
    tapa::ostream<int16_vec_t>&    out
) {
    const int total = N * HIDDEN_VECS;
    for (int i_req = 0, i_resp = 0; i_resp < total;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=917504
        if ((i_req < total) & !x_mem.read_addr.full()) {
            x_mem.read_addr.try_write(i_req); ++i_req;
        }
        if (!x_mem.read_data.empty()) {
            int16_vec_t xv; x_mem.read_data.try_read(xv);
            out.write(xv);
            ++i_resp;
        }
    }
}

// ============================================================================
// 读取任务：w_gate_reader
//   将 INT16 门控权重矩阵完整流式读取一次（权重驻留）。
//   计算任务将其缓存在片上 URAM 中，供所有 token 复用。
//   不执行计算。
// ============================================================================
void w_gate_reader(
    tapa::async_mmap<int16_vec_t>& w_gate_mem,
    tapa::ostream<int16_vec_t>&    out
) {
    for (int i_req = 0, i_resp = 0; i_resp < W_GATE_VECS;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=57344 max=57344
        if ((i_req < W_GATE_VECS) & !w_gate_mem.read_addr.full()) {
            w_gate_mem.read_addr.try_write(i_req); ++i_req;
        }
        if (!w_gate_mem.read_data.empty()) {
            int16_vec_t wv; w_gate_mem.read_data.try_read(wv);
            out.write(wv);
            ++i_resp;
        }
    }
}

// ============================================================================
// 读取任务：bias_dequant_reader
//   从单个 HBM 端口流式读取合并的 [bias | dequant] FP32 缓冲区，
//   按索引分流各次响应：索引 [0, N_EXPERTS_TOTAL) 的数据发往 bias_out，
//   索引 [N_EXPERTS_TOTAL, total) 的数据发往 dequant_out。不执行计算。
// ============================================================================
// 使用单条反量化参数流（此处不广播）。6 个浮点反量化参数沿
// expert_ffn_compute 任务链转发：每条通路读取这些参数，
// 在本地缓存后转发给下一条通路。最后一条通路的
// 下游 FIFO 由 dequant_sink 排空。
void bias_dequant_reader(
    tapa::async_mmap<float>& mem,       // 合并布局：[bias(256) | dequant(6)]
    tapa::ostream<float>&    bias_out,
    tapa::ostream<float>&    dequant_out
) {
    const int total = N_EXPERTS_TOTAL + DEQUANT_PARAMS_PER_SLOT;
    for (int i_req = 0, i_resp = 0; i_resp < total;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=262 max=262
        if ((i_req < total) & !mem.read_addr.full()) {
            mem.read_addr.try_write(i_req); ++i_req;
        }
        if (!mem.read_data.empty()) {
            float v; mem.read_data.try_read(v);
            if (i_resp < N_EXPERTS_TOTAL)
                bias_out.write(v);
            else
                dequant_out.write(v);
            ++i_resp;
        }
    }
}

// 排空任务：消费链中最后一条 expert_ffn_compute 通路转发的
// DEQUANT_PARAMS_PER_SLOT 个浮点数。若没有此任务，最后一条
// 通路的输出 FIFO 将无人读取，TAPA 因此无法终止。
void dequant_sink(
    tapa::istream<float>& dequant_in
) {
    for (int i = 0; i < DEQUANT_PARAMS_PER_SLOT; i++) {
#pragma HLS pipeline II=1
        float tmp = dequant_in.read();
        (void)tmp;
    }
}

// ============================================================================
// 计算任务：gate_compute（INT16 乘加 + 反量化）
//   纯计算，不使用 mmap。消费流式输入的 INT16 x、INT16 W_gate
//   以及 bias（仅在启动时流式读取一次）。
//   通过 INT16 乘加和反量化为全部专家评分（使用全局 x_scale、w_gate_scale），
//   执行简化的双 FPGA Top-K 路由（每半区选 4 个），并为每个
//   token 输出一个 route_pkt_t。同时通过 x_out 将 INT16 x 流转发给 dispatch_compute，
//   因此 x 只需由 x_gate_reader 从 HBM 读取一次。
// ============================================================================
void gate_compute(
    const int   N,
    const int   local_expert_base,
    const float x_scale,
    const float w_gate_scale,
    const float w_gate_shift,
    tapa::istream<int16_vec_t>&    x_in,
    tapa::istream<int16_vec_t>&    w_gate_in,
    tapa::istream<float>&          bias_in,
    tapa::ostream<route_pkt_t>&    route_out,
    tapa::ostream<int16_vec_t>&    x_out
) {
    // 仅执行一次：从 bias_reader 流读取并缓存偏置向量。
    float bias[N_EXPERTS_TOTAL];
#pragma HLS bind_storage variable=bias type=RAM_1P impl=BRAM
    for (int i = 0; i < N_EXPERTS_TOTAL; i++) {
#pragma HLS pipeline II=1
        bias[i] = bias_in.read();
    }
    // 权重驻留：将 INT16 门控权重矩阵一次性加载到片上 URAM。
    ap_uint<64> W_gate_cache[8][N_EXPERTS_TOTAL][HIDDEN_VECS];
#pragma HLS bind_storage variable=W_gate_cache type=RAM_2P impl=URAM
#pragma HLS array_partition variable=W_gate_cache complete dim=1

    for (int e = 0; e < N_EXPERTS_TOTAL; e++) {
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
            auto w_vec = w_gate_in.read();
            for(int k = 0; k < 8; k++) {
                ap_uint<64> w_val = 0;
                for(int l = 0; l < 4; l++) {
                    w_val(l*16+15, l*16) = tapa::bit_cast<ap_uint<16>>(w_vec[k*4+l]);
                }
                W_gate_cache[k][e][v] = w_val;
            }
        }
    }

    const float joint_scale = x_scale * w_gate_scale;

    for (int t = 0; t < N; t++) {
#pragma HLS loop_tripcount min=1 max=4096
        // 将当前 token 的 HIDDEN_VECS 个 INT16 向量读入 x_buf，
        // 同时通过 x_out 转发给 dispatch_compute。
        int16_vec_t x_buf[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_buf type=RAM_2P impl=BRAM
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            int16_vec_t xv = x_in.read();
            x_buf[v] = xv;
            x_out.write(xv);
        }
        // 为全部 N_EXPERTS_TOTAL 个专家评分：INT16 乘加 → 反量化 → sigmoid + 偏置。
        float scores[N_EXPERTS_TOTAL];
#pragma HLS bind_storage variable=scores type=RAM_1P impl=BRAM
        for (int e = 0; e < N_EXPERTS_TOTAL; e++) {
#pragma HLS loop_tripcount min=256 max=256
            ap_int<44> sum = 0;
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
                int16_vec_t xv = x_buf[v];
                ap_int<37> p = 0;
                for (int k = 0; k < 8; k++) {
                    for (int l = 0; l < 4; l++) {
                        ap_int<16> x_val = xv[k*4+l];
                        ap_int<16> w_val = ap_int<16>(W_gate_cache[k][e][v]((l*16)+15, l*16));
                        p += x_val * w_val;
                    }
                }
                sum += p;
            }
            scores[e] = moe_sigmoidf((float)sum * joint_scale + w_gate_shift) + bias[e];
        }
        // ----------------------------------------------------------------
        // 对每个半区并行选择前 LOCAL_K 个专家，方式与
        // indexer_vanilla.h 中的 topk_parallel_cmp 对应。用一个融合循环
        // 同步遍历两个半区，每轮分别处理每个半区的 LOCAL_K 个
        // 候选项：
        //   * 初始化：将每个半区的前 LOCAL_K 个分数填入
        //     Top-K 缓冲区；通过并行二叉归约获得初始
        //     (min_score, min_idx)。
        //   * 后续每轮对各半区的 fast_check 条件做逻辑或归约，
        //     判断是否有候选分数大于当前 min_score。若命中，则进入
        //     使用 #pragma HLS pipeline off 的内层循环，遍历 LOCAL_K 个
        //     候选项，逐个替换 top[min_idx]，并通过
        //     并行二叉归约更新最小值。
        // 低编号半区的专家范围为 [0, LOCAL_EXPERTS)；高编号半区为
        // [LOCAL_EXPERTS, N_EXPERTS_TOTAL)。
        // ----------------------------------------------------------------
        float top_lo_w [LOCAL_K];
        int   top_lo_ids[LOCAL_K];
#pragma HLS array_partition variable=top_lo_w  complete
#pragma HLS array_partition variable=top_lo_ids complete
        float top_hi_w [LOCAL_K];
        int   top_hi_ids[LOCAL_K];
#pragma HLS array_partition variable=top_hi_w  complete
#pragma HLS array_partition variable=top_hi_ids complete

        // 用每个半区的前 LOCAL_K 个分数初始化该半区。
        for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
            top_lo_w  [i] = scores[i];
            top_lo_ids[i] = i;
            top_hi_w  [i] = scores[LOCAL_EXPERTS + i];
            top_hi_ids[i] = LOCAL_EXPERTS + i;
        }
        float lo_min_s; int lo_min_i;
        float hi_min_s; int hi_min_i;
        {
            float sl01, sl23, sh01, sh23; int il01, il23, ih01, ih23;
            if (top_lo_w[0] < top_lo_w[1]) { sl01 = top_lo_w[0]; il01 = 0; } else { sl01 = top_lo_w[1]; il01 = 1; }
            if (top_lo_w[2] < top_lo_w[3]) { sl23 = top_lo_w[2]; il23 = 2; } else { sl23 = top_lo_w[3]; il23 = 3; }
            if (sl01 < sl23)              { lo_min_s = sl01; lo_min_i = il01; } else { lo_min_s = sl23; lo_min_i = il23; }
            if (top_hi_w[0] < top_hi_w[1]) { sh01 = top_hi_w[0]; ih01 = 0; } else { sh01 = top_hi_w[1]; ih01 = 1; }
            if (top_hi_w[2] < top_hi_w[3]) { sh23 = top_hi_w[2]; ih23 = 2; } else { sh23 = top_hi_w[3]; ih23 = 3; }
            if (sh01 < sh23)              { hi_min_s = sh01; hi_min_i = ih01; } else { hi_min_s = sh23; hi_min_i = ih23; }
        }
        // 融合扫描：执行 (LOCAL_EXPERTS / LOCAL_K) - 1 轮，每轮每半区处理 4 个
        // 候选项（从各半区内偏移 LOCAL_K 的位置开始）。
        for (int r = 1; r < (LOCAL_EXPERTS / LOCAL_K); r++) {
#pragma HLS loop_tripcount min=31 max=31
            float cand_lo_s[LOCAL_K];
            int   cand_lo_e[LOCAL_K];
            float cand_hi_s[LOCAL_K];
            int   cand_hi_e[LOCAL_K];
#pragma HLS array_partition variable=cand_lo_s complete
#pragma HLS array_partition variable=cand_lo_e complete
#pragma HLS array_partition variable=cand_hi_s complete
#pragma HLS array_partition variable=cand_hi_e complete
            for (int j = 0; j < LOCAL_K; j++) {
#pragma HLS unroll
                int elo = r * LOCAL_K + j;
                int ehi = LOCAL_EXPERTS + r * LOCAL_K + j;
                cand_lo_s[j] = scores[elo]; cand_lo_e[j] = elo;
                cand_hi_s[j] = scores[ehi]; cand_hi_e[j] = ehi;
            }
            bool fast_lo = false, fast_hi = false;
            for (int j = 0; j < LOCAL_K; j++) {
#pragma HLS unroll
                fast_lo |= (cand_lo_s[j] > lo_min_s);
                fast_hi |= (cand_hi_s[j] > hi_min_s);
            }
            if (fast_lo) {
                for (int j = 0; j < LOCAL_K; j++) {
#pragma HLS pipeline off
                    if (cand_lo_s[j] > lo_min_s) {
                        top_lo_w  [lo_min_i] = cand_lo_s[j];
                        top_lo_ids[lo_min_i] = cand_lo_e[j];
                        float s01, s23; int i01, i23;
                        if (top_lo_w[0] < top_lo_w[1]) { s01 = top_lo_w[0]; i01 = 0; } else { s01 = top_lo_w[1]; i01 = 1; }
                        if (top_lo_w[2] < top_lo_w[3]) { s23 = top_lo_w[2]; i23 = 2; } else { s23 = top_lo_w[3]; i23 = 3; }
                        if (s01 < s23) { lo_min_s = s01; lo_min_i = i01; } else { lo_min_s = s23; lo_min_i = i23; }
                    }
                }
            }
            if (fast_hi) {
                for (int j = 0; j < LOCAL_K; j++) {
#pragma HLS pipeline off
                    if (cand_hi_s[j] > hi_min_s) {
                        top_hi_w  [hi_min_i] = cand_hi_s[j];
                        top_hi_ids[hi_min_i] = cand_hi_e[j];
                        float s01, s23; int i01, i23;
                        if (top_hi_w[0] < top_hi_w[1]) { s01 = top_hi_w[0]; i01 = 0; } else { s01 = top_hi_w[1]; i01 = 1; }
                        if (top_hi_w[2] < top_hi_w[3]) { s23 = top_hi_w[2]; i23 = 2; } else { s23 = top_hi_w[3]; i23 = 3; }
                        if (s01 < s23) { hi_min_s = s01; hi_min_i = i01; } else { hi_min_s = s23; hi_min_i = i23; }
                    }
                }
            }
        }

        // 使用全部 8 个选中分数归一化（对两个半区的选中分数求和）。
        float wsum = 0.0f;
        for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
            wsum += top_lo_w[i] + top_hi_w[i];
        }
        float wscale = (wsum > 1e-12f) ? (ROUTE_SCALE / wsum) : 0.0f;

        // 根据本卡的 local_expert_base 输出本地前 LOCAL_K 个专家。
        // token 按顺序输出；消费者根据自己的循环索引推断 token_id，
        // 因此无需 token_id/num_local 字段。
        route_pkt_t pkt;
        if (local_expert_base == 0) {
            for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
                pkt.local_expert_ids[i] = (expert_id_t)top_lo_ids[i];
                pkt.weights[i]          = top_lo_w[i] * wscale;
            }
        } else {
            for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
                pkt.local_expert_ids[i] = (expert_id_t)top_hi_ids[i];
                pkt.weights[i]          = top_hi_w[i] * wscale;
            }
        }
        route_out.write(pkt);
    }
}

// ============================================================================
// 计算任务：dispatch_compute
//   纯计算（无 async_mmap）。消费 gate_compute 的路由包
//   和转发的 INT16 x（不访问 HBM；输入经 gate_compute
//   单路转发）。由于 x 已是 INT16，且使用全局
//   x_scale，无需逐 token 量化；此任务只负责广播：
//   每个 token 的 INT16 x 缓存一次，随后输出 LOCAL_K 组
//   数据，每组包含 hdr 和 HIDDEN_VECS 个 INT16 向量。
// ============================================================================
void dispatch_compute(
    const int   N,
    const int   local_expert_base,
    const float x_scale,
    tapa::istream<route_pkt_t>&                          route_in,
    tapa::istream<int16_vec_t>&                          x_in,
    tapa::ostreams<dispatch_hdr_t, LOCAL_K>&             hdr_out,
    tapa::ostreams<int16_vec_t,    LOCAL_K>&             x_out,
    // 广播触发信号：每个（矩阵，通路）对应一个信号。直接发送给权重
    // 加载器，使其能够异步预取权重，与
    // expert_ffn_compute 的流水线解耦。
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger1_out,
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger3_out,
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger2_out,
    // 优化 3：奇数行子矩阵加载器的触发信号。
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger1b_out,
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger3b_out,
    tapa::ostreams<weight_trigger_t, LOCAL_K>&           trigger2b_out
) {
    for (int t = 0; t < N; t++) {
#pragma HLS loop_tripcount min=1 max=4096

        // 缓存当前 token 的 INT16 激活，以便同时重放给全部
        // LOCAL_K 条并行专家通路。
        int16_vec_t x_int_buf[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_int_buf type=RAM_2P impl=BRAM
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            x_int_buf[v] = x_in.read();
        }

        route_pkt_t route = route_in.read();

        // 每条并行通路发送一个头部，并写入三个触发信号（每个矩阵
        // 加载器一个）。触发信号由分发任务直接送达加载器，
        // 使加载器无需等待当前 token 计算完成，
        // 即可发起下一次预取。
        for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
            int eid_global = (int)route.local_expert_ids[i];
            int rel_eid    = eid_global - local_expert_base;
            dispatch_hdr_t hdr;
            hdr.token_id         = (token_id_t)t;
            hdr.global_expert_id = (expert_id_t)eid_global;
            hdr.routing_weight   = route.weights[i];
            hdr.x_scale          = x_scale;
            hdr.done             = false;
            hdr_out[i].write(hdr);
            trigger1_out[i].write(rel_eid);
            trigger3_out[i].write(rel_eid);
            trigger2_out[i].write(rel_eid);
            trigger1b_out[i].write(rel_eid);
            trigger3b_out[i].write(rel_eid);
            trigger2b_out[i].write(rel_eid);
        }

        // 将激活张量同步复制到全部 LOCAL_K 条通路。
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            int16_vec_t xv = x_int_buf[v];
            for (int i = 0; i < LOCAL_K; i++) {
#pragma HLS unroll
                x_out[i].write(xv);
            }
        }
    }
}

// ============================================================================
// 读取任务：w1/w3/w2_weight_loader
//   每个矩阵使用一个 HBM 端口。每个加载器消费本轮的触发信号
//   （相对专家编号），然后将矩阵数据逐拍流式写入专用的
//   下游 FIFO。每条通路有三个独立端口，因此 W1 和 W3 可
//   并行读取（供融合乘加循环使用），同时 W2 可在
//   第一阶段运行期间预取。
// ============================================================================
void w1_weight_loader(
    const int N,
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W1_mem,
    tapa::ostream<int16_vec_t>&      W1_fifo
) {
    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        weight_trigger_t trig = trigger_in.read();
        const int base = trig * W1_HALF_VECS;
        for (int i_req = 0, i_resp = 0; i_resp < W1_HALF_VECS;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=65536 max=65536
            if ((i_req < W1_HALF_VECS) & !W1_mem.read_addr.full()) {
                W1_mem.read_addr.try_write(base + i_req); ++i_req;
            }
            if (!W1_mem.read_data.empty()) {
                int16_vec_t tmp; W1_mem.read_data.try_read(tmp);
                W1_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

void w3_weight_loader(
    const int N,
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W3_mem,
    tapa::ostream<int16_vec_t>&      W3_fifo
) {
    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        weight_trigger_t trig = trigger_in.read();
        const int base = trig * W3_HALF_VECS;
        for (int i_req = 0, i_resp = 0; i_resp < W3_HALF_VECS;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=65536 max=65536
            if ((i_req < W3_HALF_VECS) & !W3_mem.read_addr.full()) {
                W3_mem.read_addr.try_write(base + i_req); ++i_req;
            }
            if (!W3_mem.read_data.empty()) {
                int16_vec_t tmp; W3_mem.read_data.try_read(tmp);
                W3_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

void w2_weight_loader(
    const int N,
    tapa::istream<weight_trigger_t>& trigger_in,
    tapa::async_mmap<int16_vec_t>&   W2_mem,
    tapa::ostream<int16_vec_t>&      W2_fifo
) {
    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        weight_trigger_t trig = trigger_in.read();
        const int base = trig * W2_HALF_VECS;
        for (int i_req = 0, i_resp = 0; i_resp < W2_HALF_VECS;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=229376 max=229376
            if ((i_req < W2_HALF_VECS) & !W2_mem.read_addr.full()) {
                W2_mem.read_addr.try_write(base + i_req); ++i_req;
            }
            if (!W2_mem.read_data.empty()) {
                int16_vec_t tmp; W2_mem.read_data.try_read(tmp);
                W2_fifo.write(tmp);
                ++i_resp;
            }
        }
    }
}

// ============================================================================
// 优化 2：将每条通路的 FFN 数据流拆成 6 个协作任务。
//
//   dequant_split  → 将包含 6 个浮点数的反量化参数链分发为各投影的参数对
//   ffn_x_split    → 将 hdr 标量和 INT16 激活广播给 W1/W3 通路
//   ffn_w1_proj    → W1 矩阵向量乘法 + 反量化 → MOE_INTER 个 FP32 acc1
//   ffn_w3_proj    → W3 矩阵向量乘法 + 反量化 → MOE_INTER 个 FP32 acc3
//   ffn_swiglu     → SiLU(acc1)*acc3，再量化 → INT16 中间向量
//   ffn_w2_proj    → W2 矩阵向量乘法 + 反量化 + 按路由权重累加部分和
// ============================================================================

// dequant_split：沿链转发全部 6 个浮点反量化参数，同时
// 将各矩阵的 (scale, shift) 参数对发给本通路的投影任务。
//   索引 0、1 → w1_dq_out  (w1_scale, w1_shift)
//   索引 2、3 → w3_dq_out  (w3_scale, w3_shift)
//   索引 4、5 → w2_dq_out  (w2_scale, w2_shift)
void dequant_split(
    tapa::istream<float>& dequant_in,
    tapa::ostream<float>& dequant_fwd,
    tapa::ostream<float>& w1_dq_out,
    tapa::ostream<float>& w3_dq_out,
    tapa::ostream<float>& w2_dq_out
) {
    for (int i = 0; i < DEQUANT_PARAMS_PER_SLOT; i++) {
#pragma HLS pipeline II=1
        float v = dequant_in.read();
        dequant_fwd.write(v);
        if (i == 0 || i == 1) {
            w1_dq_out.write(v);
        } else if (i == 2 || i == 3) {
            w3_dq_out.write(v);
        } else {
            w2_dq_out.write(v);
        }
    }
}

// ffn_x_split：将分发头部中每个 token 的标量广播给
// 下游投影任务，并将 INT16 激活张量复制给
// W1 和 W3 通路（W2 仅需路由权重，该权重通过独立流
// 为每个 token 转发一次）。
void ffn_x_split(
    const int N,
    tapa::istream<dispatch_hdr_t>& hdr_in,
    tapa::istream<int16_vec_t>&    x_in,
    tapa::ostream<int16_vec_t>&    x1_out,
    tapa::ostream<int16_vec_t>&    x3_out,
    tapa::ostream<float>&          xs_w1_out,
    tapa::ostream<float>&          xs_w3_out,
    tapa::ostream<float>&          w_route_out
) {
    for (int t = 0; t < N; t++) {
#pragma HLS loop_tripcount min=1 max=4096
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            int16_vec_t xv = x_in.read();
            x1_out.write(xv);
            x3_out.write(xv);
        }
        dispatch_hdr_t hdr = hdr_in.read();
        xs_w1_out.write(hdr.x_scale);
        xs_w3_out.write(hdr.x_scale);
        w_route_out.write(hdr.routing_weight);
    }
}

// ffn_w1_proj：W1 矩阵向量乘法 + 反量化，采用两路输出行并行（优化 3）。
// 每轮内层迭代同时处理 W1a_fifo 的偶数行和 W1b_fifo 的奇数行。
// 每个输出拍将两个结果打包为 float2_t（每周期只写一次 FIFO，
// 携带两个 FP32 值，避免同周期两次写入冲突）。
void ffn_w1_proj(
    const int N,
    tapa::istream<int16_vec_t>& x1_in,
    tapa::istream<int16_vec_t>& W1a_fifo,
    tapa::istream<int16_vec_t>& W1b_fifo,
    tapa::istream<float>&       xs_w1_in,
    tapa::istream<float>&       w1_dq_in,
    tapa::ostream<float2_t>&    acc1_out
) {
    float w1_scale = w1_dq_in.read();
    float w1_shift = w1_dq_in.read();

    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096

        int16_vec_t x_cur[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_cur type=RAM_2P impl=BRAM
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            x_cur[v] = x1_in.read();
        }
        float x_scale = xs_w1_in.read();
        float scale_xw = w1_scale * x_scale;
        for (int rh = 0; rh < MOE_INTER/2; rh++) {
#pragma HLS loop_tripcount min=1024 max=1024
            ap_int<44> sum_a = 0, sum_b = 0;
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
                int16_vec_t xv   = x_cur[v];
                int16_vec_t w1av = W1a_fifo.read();
                int16_vec_t w1bv = W1b_fifo.read();
                ap_int<37> pa = 0, pb = 0;
                for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                    pa += xv[k] * w1av[k];
                    pb += xv[k] * w1bv[k];
                }
                sum_a += pa;
                sum_b += pb;
            }
            float2_t pair;
            pair[0] = (float)sum_a * scale_xw + w1_shift;
            pair[1] = (float)sum_b * scale_xw + w1_shift;
            acc1_out.write(pair);
        }
    }
}

// ffn_w3_proj：与 ffn_w1_proj 结构对称，采用两路输出行并行。
void ffn_w3_proj(
    const int N,
    tapa::istream<int16_vec_t>& x3_in,
    tapa::istream<int16_vec_t>& W3a_fifo,
    tapa::istream<int16_vec_t>& W3b_fifo,
    tapa::istream<float>&       xs_w3_in,
    tapa::istream<float>&       w3_dq_in,
    tapa::ostream<float2_t>&    acc3_out
) {
    float w3_scale = w3_dq_in.read();
    float w3_shift = w3_dq_in.read();

    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096

        int16_vec_t x_cur[HIDDEN_VECS];
#pragma HLS bind_storage variable=x_cur type=RAM_2P impl=BRAM
        for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
            x_cur[v] = x3_in.read();
        }

        float x_scale = xs_w3_in.read();
        float scale_xw = w3_scale * x_scale;
        for (int rh = 0; rh < MOE_INTER/2; rh++) {
#pragma HLS loop_tripcount min=1024 max=1024
            ap_int<44> sum_a = 0, sum_b = 0;
            for (int v = 0; v < HIDDEN_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=224 max=224
                int16_vec_t xv   = x_cur[v];
                int16_vec_t w3av = W3a_fifo.read();
                int16_vec_t w3bv = W3b_fifo.read();
                ap_int<37> pa = 0, pb = 0;
                for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                    pa += xv[k] * w3av[k];
                    pb += xv[k] * w3bv[k];
                }
                sum_a += pa;
                sum_b += pb;
            }
            float2_t pair;
            pair[0] = (float)sum_a * scale_xw + w3_shift;
            pair[1] = (float)sum_b * scale_xw + w3_shift;
            acc3_out.write(pair);
        }
    }
}

// ffn_swiglu：SiLU 门控 + 再量化（优化 3：消费 float2_t 数据对）。
// acc1_in / acc3_in 每拍携带两个 FP32 值（偶数行和奇数行），因此
// 门控循环执行 MOE_INTER/2 轮，每周期向 inter_buf 填入两个元素，
// 无需在同一周期内对任一输入 FIFO 读取两次。
void ffn_swiglu(
    const int N,
    tapa::istream<float2_t>&    acc1_in,
    tapa::istream<float2_t>&    acc3_in,
    tapa::ostream<int16_vec_t>& inter_out,
    tapa::ostream<float>&       inter_scale_out
) {
    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        float inter_buf[MOE_INTER];
#pragma HLS bind_storage variable=inter_buf type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=inter_buf cyclic factor=32

        float inter_max = 0.0f;
        for (int rh = 0; rh < MOE_INTER/2; rh++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=1024 max=1024
            float2_t p1 = acc1_in.read();
            float2_t p3 = acc3_in.read();
            float va = moe_siluf(p1[0]) * p3[0];
            float vb = moe_siluf(p1[1]) * p3[1];
            inter_buf[2*rh]   = va;
            inter_buf[2*rh+1] = vb;
            float aa = (va >= 0.0f) ? va : -va;
            float ab = (vb >= 0.0f) ? vb : -vb;
            float mx = (aa > ab) ? aa : ab;
            if (mx > inter_max) inter_max = mx;
        }

        float inter_scale = (inter_max > 0.0f) ? (inter_max / 32767.0f) : 1.0f;
        float inter_inv   = 1.0f / inter_scale;

        inter_scale_out.write(inter_scale);

        for (int v = 0; v < INTER_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64
            int16_vec_t iv;
            for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                float f = inter_buf[v * VEC_WIDTH + k] * inter_inv;
                int q = (int)(f >= 0.0f ? f + 0.5f : f - 0.5f);
                if (q >  32767) q =  32767;
                if (q < -32768) q = -32768;
                iv[k] = (ap_int<16>)q;
            }
            inter_out.write(iv);
        }
    }
}

// ffn_w2_proj：W2 矩阵向量乘法 + 反量化 + 部分和归约，采用两路行并行。
// 每轮内层迭代读取 W2a_fifo 的偶数行和 W2b_fifo 的奇数行；
// 每轮外层迭代填入两个 y_buf 元素，使 W2 延迟减半。
void ffn_w2_proj(
    const int N,
    tapa::istream<int16_vec_t>& inter_in,
    tapa::istream<float>&       inter_scale_in,
    tapa::istream<float>&       w_route_in,
    tapa::istream<int16_vec_t>& W2a_fifo,
    tapa::istream<int16_vec_t>& W2b_fifo,
    tapa::istream<float>&       w2_dq_in,
    tapa::istream<fp32_vec_t>&  partial_in,
    tapa::ostream<fp32_vec_t>&  partial_out
) {
    float w2_scale = w2_dq_in.read();
    float w2_shift = w2_dq_in.read();

    for (int iter = 0; iter < N; iter++) {
#pragma HLS loop_tripcount min=1 max=4096
        float inter_scale = inter_scale_in.read();
        float w_route     = w_route_in.read();

        int16_vec_t inter_buf[INTER_VECS];
#pragma HLS bind_storage variable=inter_buf type=RAM_2P impl=BRAM
        for (int v = 0; v < INTER_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64
            inter_buf[v] = inter_in.read();
        }

        float y_buf[HIDDEN];
#pragma HLS bind_storage variable=y_buf type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=y_buf cyclic factor=16

        float scale_wi = w2_scale * inter_scale;
        for (int rh = 0; rh < HIDDEN/2; rh++) {
#pragma HLS loop_tripcount min=3584 max=3584
            ap_int<43> sum_a = 0, sum_b = 0;
            for (int v = 0; v < INTER_VECS; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=64 max=64
                int16_vec_t iv   = inter_buf[v];
                int16_vec_t w2av = W2a_fifo.read();
                int16_vec_t w2bv = W2b_fifo.read();
                ap_int<37> pa = 0, pb = 0;
                for (int k = 0; k < VEC_WIDTH; k++) {
#pragma HLS unroll
                    pa += iv[k] * w2av[k];
                    pb += iv[k] * w2bv[k];
                }
                sum_a += pa;
                sum_b += pb;
            }
            y_buf[2*rh]   = (float)sum_a * scale_wi + w2_shift;
            y_buf[2*rh+1] = (float)sum_b * scale_wi + w2_shift;
        }

        for (int v = 0; v < HIDDEN_VECS_FP; v++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=448 max=448
            fp32_vec_t pin = partial_in.read();
            fp32_vec_t pout;
            for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
                pout[k] = pin[k] + w_route * y_buf[v * VEC_WIDTH_FP + k];
            }
            partial_out.write(pout);
        }
    }
}

// ============================================================================
// 计算任务：partial_seed
//   用 N * HIDDEN_VECS_FP 个全零 FP32 向量初始化流式归约链。
//   专家链的通路 0 将这些向量作为 partial_in 读取，
//   并将自身贡献作为 partial_out 输出；后续通路读取
//   当前累加和，再加上自身的加权贡献。最后一条通路输出
//   完成累加的 y，直接送入 y_writer。
// ============================================================================
void partial_seed(
    const int N,
    tapa::ostream<fp32_vec_t>& seed_out
) {
    const int total = N * HIDDEN_VECS_FP;
    for (int i = 0; i < total; i++) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=448 max=1835008
        fp32_vec_t z;
        for (int k = 0; k < VEC_WIDTH_FP; k++) {
#pragma HLS unroll
            z[k] = 0.0f;
        }
        seed_out.write(z);
    }
}

// ============================================================================
// 写入任务：y_writer
//   使用 async_mmap 将 N * HIDDEN_VECS_FP 个 FP32 向量写入 y_mem，
//   使请求发送和响应接收重叠执行。不执行计算。
// ============================================================================
void y_writer(
    const int N,
    tapa::istream<fp32_vec_t>&      y_in,
    tapa::async_mmap<fp32_vec_t>&   y_mem
) {
    const int total = N * HIDDEN_VECS_FP;

    for (int i_req = 0, i_resp = 0; i_resp < total;) {
#pragma HLS pipeline II=1
#pragma HLS loop_tripcount min=448 max=28672
        if ((i_req < total) & !y_mem.write_addr.full() & !y_mem.write_data.full()
            & !y_in.empty()) {
            fp32_vec_t ov; y_in.try_read(ov);
            y_mem.write_addr.try_write(i_req);
            y_mem.write_data.try_write(ov);
            ++i_req;
        }
        if (!y_mem.write_resp.empty()) {
            bool ok = false;
            uint8_t resp = y_mem.write_resp.read(ok);
            if (ok) i_resp += (int)resp + 1;
        }
    }
}

// ============================================================================
// 顶层任务图
//
// 读取任务（仅 async_mmap） → 计算任务（仅使用流） → 写入任务（仅 async_mmap）
// ============================================================================
void moe_fpga_top(
    const int   N,
    const int   local_expert_base,
    const float x_scale,
    const float w_gate_scale,
    const float w_gate_shift,
    tapa::mmap<int16_vec_t> x_mem,
    tapa::mmap<int16_vec_t> W_gate_mem,
    tapa::mmap<float>                              bias_dequant_mem,
    // 优化 3：每条并行通路使用六个 HBM 端口，分别对应偶数行和奇数行子矩阵。
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W1_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W1b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W3_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W3b_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W2_expert_mem,
    tapa::mmaps<int16_vec_t, N_PARALLEL_SLOTS>     W2b_expert_mem,
    tapa::mmap<fp32_vec_t>                         y_mem
) {
    // 读取任务 → gate_compute
    tapa::stream<int16_vec_t> x_gate_fifo("x_gate_fifo");
    tapa::stream<int16_vec_t> w_gate_fifo("w_gate_fifo");
    tapa::stream<float> bias_fifo  ("bias_fifo");

    // 反量化参数链：bias_dequant_reader → dequant_split[0] → ... → dequant_split[K-1] → 排空任务。
    // 深度 8 足以容纳全部 DEQUANT_PARAMS_PER_SLOT（=6）个元素。
    tapa::streams<float, N_PARALLEL_SLOTS + 1, 8> dequant_chain("dequant_chain");

    // dequant_split 的各通路输出（向每个投影发送 2 个浮点参数）。
    tapa::streams<float, N_PARALLEL_SLOTS, 4> w1_dq_fifos("w1_dq");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> w3_dq_fifos("w3_dq");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> w2_dq_fifos("w2_dq");

    // gate_compute → dispatch_compute
    tapa::stream<route_pkt_t,   4>  route_out  ("route_out");
    tapa::stream<int16_vec_t, 4>  x_disp_fifo("x_disp_fifo");

    // dispatch_compute → ffn_x_split（每条并行通路一条流）
    tapa::streams<dispatch_hdr_t, N_PARALLEL_SLOTS, 4>   disp_hdr_fifos("disp_hdr");
    tapa::streams<int16_vec_t,    N_PARALLEL_SLOTS, 4> disp_x_fifos  ("disp_x");

    // dispatch_compute → 权重加载器（每条通路的每个矩阵各一条流）。
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger1_fifos("trigger1");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger3_fifos("trigger3");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger2_fifos("trigger2");
    // 优化 3：奇数行子矩阵加载器的触发信号。
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger1b_fifos("trigger1b");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger3b_fifos("trigger3b");
    tapa::streams<weight_trigger_t, N_PARALLEL_SLOTS, 4> trigger2b_fifos("trigger2b");

    // 权重加载器 → 投影任务（每条通路的每个矩阵各一条流）。
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W1_fifos("W1_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W3_fifos("W3_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W2_fifos("W2_fifo");
    // 优化 3：奇数行子矩阵的 FIFO。
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W1b_fifos("W1b_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W3b_fifos("W3b_fifo");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS> W2b_fifos("W2b_fifo");

    // ffn_x_split → ffn_w1_proj / ffn_w3_proj / ffn_w2_proj.
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 4> x1_fifos("x1");
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS, 4> x3_fifos("x3");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> xs_w1_fifos  ("xs_w1");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> xs_w3_fifos  ("xs_w3");
    tapa::streams<float, N_PARALLEL_SLOTS, 4> w_route_fifos("w_route");

    // 投影 → swiglu：每拍用 float2_t 打包偶数行、奇数行的结果。
    tapa::streams<float2_t, N_PARALLEL_SLOTS, 4> acc1_fifos("acc1");
    tapa::streams<float2_t, N_PARALLEL_SLOTS, 4> acc3_fifos("acc3");

    // swiglu → ffn_w2_proj.
    tapa::streams<int16_vec_t, N_PARALLEL_SLOTS,  4> inter_fifos      ("inter");
    tapa::streams<float,       N_PARALLEL_SLOTS,  4> inter_scale_fifos("inter_scale");

    // 流式归约链：初始零向量 → 通路 0 → 通路 1 → ... → 通路 K-1 → y_writer。
    tapa::streams<fp32_vec_t, N_PARALLEL_SLOTS + 1, 16> partial_chain("partial_chain");

    tapa::task()
        .invoke<tapa::join>(x_gate_reader,       N, x_mem, x_gate_fifo)
        .invoke<tapa::join>(w_gate_reader,       W_gate_mem, w_gate_fifo)
        .invoke<tapa::join>(bias_dequant_reader, bias_dequant_mem,
                            bias_fifo, dequant_chain)
        .invoke<tapa::join>(gate_compute,        N, local_expert_base,
                            x_scale, w_gate_scale, w_gate_shift,
                            x_gate_fifo, w_gate_fifo, bias_fifo,
                            route_out, x_disp_fifo)
        .invoke<tapa::join>(dispatch_compute,    N, local_expert_base, x_scale,
                            route_out, x_disp_fifo,
                            disp_hdr_fifos, disp_x_fifos,
                            trigger1_fifos, trigger3_fifos, trigger2_fifos,
                            trigger1b_fifos, trigger3b_fifos, trigger2b_fifos)
        // dequant_split 链：实例 i 读取 dequant_chain[i]，写入
        // dequant_chain[i+1]，并输出本通路的 (w1/w3/w2) 反量化参数对。
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            dequant_split,
            dequant_chain, dequant_chain,
            w1_dq_fifos, w3_dq_fifos, w2_dq_fifos)
        .invoke<tapa::join>(dequant_sink, dequant_chain)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_x_split,
            N, disp_hdr_fifos, disp_x_fifos,
            x1_fifos, x3_fifos, xs_w1_fifos, xs_w3_fifos, w_route_fifos)
        // 偶数行加载器（使用相同函数，连接不同端口）
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w1_weight_loader, N, trigger1_fifos, W1_expert_mem, W1_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w3_weight_loader, N, trigger3_fifos, W3_expert_mem, W3_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w2_weight_loader, N, trigger2_fifos, W2_expert_mem, W2_fifos)
        // 奇数行加载器（使用相同函数，连接 b 端口）
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w1_weight_loader, N, trigger1b_fifos, W1b_expert_mem, W1b_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w3_weight_loader, N, trigger3b_fifos, W3b_expert_mem, W3b_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            w2_weight_loader, N, trigger2b_fifos, W2b_expert_mem, W2b_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w1_proj,
            N, x1_fifos, W1_fifos, W1b_fifos, xs_w1_fifos, w1_dq_fifos, acc1_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w3_proj,
            N, x3_fifos, W3_fifos, W3b_fifos, xs_w3_fifos, w3_dq_fifos, acc3_fifos)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_swiglu,
            N, acc1_fifos, acc3_fifos, inter_fifos, inter_scale_fifos)
        .invoke<tapa::join>(partial_seed, N, partial_chain)
        .invoke<tapa::join, N_PARALLEL_SLOTS>(
            ffn_w2_proj,
            N, inter_fifos, inter_scale_fifos, w_route_fifos,
            W2_fifos, W2b_fifos, w2_dq_fifos,
            partial_chain, partial_chain)
        .invoke<tapa::join>(y_writer, N, partial_chain, y_mem);
}
