#include "server-turn.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cstring>

//
// Silero VAD (16 kHz), see tools/server/realtime-models/convert_silero_vad.py
//
// each frame of 512 samples goes in after the last 64 samples of the previous one; the model is a 256-point STFT,
// four convolutions, an LSTM cell and a linear output
//

struct server_vad_model {
    gguf_context * gctx = nullptr;
    ggml_context * ctx  = nullptr;

    ggml_tensor * stft    = nullptr; // [256, 1, 258]
    ggml_tensor * conv_w[4] = {};
    ggml_tensor * conv_b[4] = {};
    ggml_tensor * w_ih    = nullptr; // [128, 512]
    ggml_tensor * w_hh    = nullptr;
    ggml_tensor * b_ih    = nullptr;
    ggml_tensor * b_hh    = nullptr;
    ggml_tensor * out_w   = nullptr; // [1, 128, 1]
    ggml_tensor * out_b   = nullptr;

    ~server_vad_model() {
        if (ctx) {
            ggml_free(ctx);
        }
        if (gctx) {
            gguf_free(gctx);
        }
    }
};

std::shared_ptr<server_vad_model> server_vad_model_load(const std::string & path) {
    auto m = std::make_shared<server_vad_model>();
    gguf_init_params params = { /*no_alloc*/ false, /*ctx*/ &m->ctx };
    m->gctx = gguf_init_from_file(path.c_str(), params);
    if (!m->gctx) {
        LOG_ERR("%s: failed to load %s\n", __func__, path.c_str());
        return nullptr;
    }
    const int64_t arch = gguf_find_key(m->gctx, "general.architecture");
    if (arch < 0 || std::string(gguf_get_val_str(m->gctx, arch)) != "silero-vad") {
        LOG_ERR("%s: %s is not a silero-vad model\n", __func__, path.c_str());
        return nullptr;
    }
    auto get = [&](const char * name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(m->ctx, name);
        if (!t) {
            LOG_ERR("%s: missing tensor %s\n", __func__, name);
        }
        return t;
    };
    m->stft = get("stft_conv.weight");
    for (int i = 0; i < 4; i++) {
        m->conv_w[i] = get(("conv" + std::to_string(i + 1) + ".weight").c_str());
        m->conv_b[i] = get(("conv" + std::to_string(i + 1) + ".bias").c_str());
    }
    m->w_ih  = get("lstm_cell.weight_ih");
    m->w_hh  = get("lstm_cell.weight_hh");
    m->b_ih  = get("lstm_cell.bias_ih");
    m->b_hh  = get("lstm_cell.bias_hh");
    m->out_w = get("final_conv.weight");
    m->out_b = get("final_conv.bias");
    for (ggml_tensor * t : {m->stft, m->w_ih, m->w_hh, m->b_ih, m->b_hh, m->out_w, m->out_b}) {
        if (!t) {
            return nullptr;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (!m->conv_w[i] || !m->conv_b[i]) {
            return nullptr;
        }
    }
    return m;
}

namespace {

constexpr int SILERO_RATE    = 16000;
constexpr int SILERO_FRAME   = 512;
constexpr int SILERO_CONTEXT = 64;
constexpr int SILERO_HIDDEN  = 128;

// conv1d with an F32 im2col: x [len, in_ch], kernel [k, in_ch, out_ch] -> [out_len, out_ch]
ggml_tensor * conv1d(ggml_context * ctx, ggml_tensor * k, ggml_tensor * b, ggml_tensor * x, int stride, int pad) {
    ggml_tensor * col = ggml_im2col(ctx, k, x, stride, 0, pad, 0, 1, 0, false, GGML_TYPE_F32);
    ggml_tensor * y   = ggml_mul_mat(ctx, ggml_reshape_2d(ctx, col, col->ne[0], col->ne[1] * col->ne[2]),
                                          ggml_reshape_2d(ctx, k, k->ne[0] * k->ne[1], k->ne[2]));
    return b ? ggml_add(ctx, y, ggml_reshape_2d(ctx, b, 1, b->ne[0])) : y;
}

struct vad_silero : server_vad {
    std::shared_ptr<server_vad_model> m;
    ggml_context *       ctx  = nullptr;
    ggml_cgraph *        gf   = nullptr;
    ggml_tensor *        inp  = nullptr; // context + frame
    ggml_tensor *        h    = nullptr;
    ggml_tensor *        c    = nullptr;
    ggml_tensor *        prob = nullptr;
    ggml_cplan           plan = {};
    std::vector<uint8_t> work;

    explicit vad_silero(std::shared_ptr<server_vad_model> model) : m(std::move(model)) {
        ggml_init_params ip = { /*mem_size*/ 8u * 1024 * 1024, /*mem_buffer*/ nullptr, /*no_alloc*/ false };
        ctx = ggml_init(ip);

        inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, SILERO_CONTEXT + SILERO_FRAME);
        h   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, SILERO_HIDDEN);
        c   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, SILERO_HIDDEN);

        // STFT magnitude: 4 frames x 129 bins
        ggml_tensor * x    = ggml_pad_reflect_1d(ctx, ggml_reshape_2d(ctx, inp, inp->ne[0], 1), 0, SILERO_CONTEXT);
        ggml_tensor * spec = conv1d(ctx, m->stft, nullptr, x, 128, 0); // [4, 258]
        const int     nb   = (int) spec->ne[1] / 2;
        ggml_tensor * re   = ggml_view_2d(ctx, spec, spec->ne[0], nb, spec->nb[1], 0);
        ggml_tensor * im   = ggml_view_2d(ctx, spec, spec->ne[0], nb, spec->nb[1], nb * spec->nb[1]);
        ggml_tensor * cur  = ggml_sqrt(ctx, ggml_add(ctx, ggml_sqr(ctx, re), ggml_sqr(ctx, im)));
        cur = ggml_cont(ctx, cur);

        const int strides[4] = {1, 2, 2, 1};
        for (int i = 0; i < 4; i++) {
            cur = ggml_relu(ctx, conv1d(ctx, m->conv_w[i], m->conv_b[i], cur, strides[i], 1)); // -> [1, 128]
        }
        ggml_tensor * xt = ggml_reshape_1d(ctx, ggml_cont(ctx, cur), SILERO_HIDDEN);

        // LSTM cell, gates in the order i, f, g, o
        ggml_tensor * gates = ggml_add(ctx, ggml_add(ctx, ggml_mul_mat(ctx, m->w_ih, xt), m->b_ih),
                                            ggml_add(ctx, ggml_mul_mat(ctx, m->w_hh, h), m->b_hh));
        const size_t  gs = ggml_row_size(gates->type, SILERO_HIDDEN);
        ggml_tensor * gi = ggml_sigmoid(ctx, ggml_view_1d(ctx, gates, SILERO_HIDDEN, 0 * gs));
        ggml_tensor * gf_ = ggml_sigmoid(ctx, ggml_view_1d(ctx, gates, SILERO_HIDDEN, 1 * gs));
        ggml_tensor * gg = ggml_tanh(ctx, ggml_view_1d(ctx, gates, SILERO_HIDDEN, 2 * gs));
        ggml_tensor * go = ggml_sigmoid(ctx, ggml_view_1d(ctx, gates, SILERO_HIDDEN, 3 * gs));
        ggml_tensor * c1 = ggml_add(ctx, ggml_mul(ctx, gf_, c), ggml_mul(ctx, gi, gg));
        ggml_tensor * h1 = ggml_mul(ctx, go, ggml_tanh(ctx, c1));

        ggml_tensor * w = ggml_reshape_2d(ctx, m->out_w, SILERO_HIDDEN, 1);
        prob = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, w, ggml_relu(ctx, h1)), m->out_b));

        gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, prob);
        // the state is written back after the output is computed from it
        ggml_build_forward_expand(gf, ggml_cpy(ctx, c1, c));
        ggml_build_forward_expand(gf, ggml_cpy(ctx, h1, h));

        plan = ggml_graph_plan(gf, 1, nullptr);
        work.resize(plan.work_size);
        plan.work_data = work.data();
        reset();
    }

    ~vad_silero() override {
        ggml_free(ctx);
    }

    int sample_rate() const override { return SILERO_RATE; }
    int frame_size() const override { return SILERO_FRAME; }

    float speech_prob(const float * frame) override {
        float * x = (float *) inp->data;
        memmove(x, x + SILERO_FRAME, SILERO_CONTEXT * sizeof(float)); // last 64 samples of the previous frame
        memcpy(x + SILERO_CONTEXT, frame, SILERO_FRAME * sizeof(float));
        if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
            return 0.0f;
        }
        return ((const float *) prob->data)[0];
    }

    void reset() override {
        memset(inp->data, 0, ggml_nbytes(inp));
        memset(h->data, 0, ggml_nbytes(h));
        memset(c->data, 0, ggml_nbytes(c));
    }
};

} // namespace

std::unique_ptr<server_vad> server_vad_silero(std::shared_ptr<server_vad_model> model) {
    return std::make_unique<vad_silero>(std::move(model));
}
