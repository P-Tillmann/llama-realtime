#include "server-turn.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "log.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <mutex>

//
// Smart Turn v3 (pipecat-ai/smart-turn, BSD-2), see tools/server/realtime-models/convert_smart_turn.py
//
// the last 8 s of the turn (front zero-padded) -> Whisper log-mel 80 x 800 -> Whisper-tiny encoder (400 positions)
// -> attention pooling -> MLP -> probability that the speaker finished the turn
//

namespace {

constexpr int ST_RATE    = 16000;
constexpr int ST_SAMPLES = 8 * ST_RATE;
constexpr int ST_FFT     = 400;
constexpr int ST_HOP     = 160;
constexpr int ST_MEL     = 80;
constexpr int ST_FRAMES  = 800;

typedef std::complex<float> cf;

// mixed-radix FFT, 400 = 2^4 * 25, the odd tail is a plain DFT
void fft_rec(const cf * tw, const float * in, int stride, cf * out, int n) {
    const int r = ST_FFT / n;
    if (n % 2 == 1) {
        for (int k = 0; k < n; k++) {
            cf acc = 0;
            for (int t = 0; t < n; t++) {
                acc += in[t * stride] * tw[(k * t % n) * r];
            }
            out[k] = acc;
        }
        return;
    }
    fft_rec(tw, in, 2 * stride, out, n / 2);
    fft_rec(tw, in + stride, 2 * stride, out + n / 2, n / 2);
    for (int k = 0; k < n / 2; k++) {
        const cf e = out[k], t = tw[k * r] * out[k + n / 2];
        out[k]         = e + t;
        out[k + n / 2] = e - t;
    }
}

} // namespace

struct smart_turn_model : server_turn_model {
    gguf_context * gctx = nullptr;
    ggml_context * wctx = nullptr; // weights
    ggml_context * ctx  = nullptr; // graph
    ggml_gallocr_t galloc = nullptr;
    ggml_cgraph *  gf   = nullptr;
    ggml_tensor *  mel  = nullptr;
    ggml_tensor *  prob = nullptr;
    int            n_threads = 1;
    int            n_layer = 4, d = 384, n_head = 6, n_ctx = 400;

    std::vector<float> filters; // [80][201]
    std::vector<cf>    tw;
    std::vector<float> hann;
    std::mutex         mtx; // one inference at a time, the sessions share the model

    ~smart_turn_model() override {
        if (galloc) {
            ggml_gallocr_free(galloc);
        }
        if (ctx) {
            ggml_free(ctx);
        }
        if (wctx) {
            ggml_free(wctx);
        }
        if (gctx) {
            gguf_free(gctx);
        }
    }

    int sample_rate() const override { return ST_RATE; }

    ggml_tensor * get(const std::string & name) const {
        ggml_tensor * t = ggml_get_tensor(wctx, name.c_str());
        if (!t) {
            throw std::runtime_error("smart-turn: missing tensor " + name);
        }
        return t;
    }

    static ggml_tensor * layer_norm(ggml_context * c, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        return ggml_add(c, ggml_mul(c, ggml_norm(c, x, 1e-5f), w), b);
    }

    ggml_tensor * linear(ggml_context * c, ggml_tensor * x, const std::string & name, bool bias = true) const {
        ggml_tensor * y = ggml_mul_mat(c, get(name + ".weight"), x);
        return bias ? ggml_add(c, y, get(name + ".bias")) : y;
    }

    static ggml_tensor * conv1d(ggml_context * c, ggml_tensor * k, ggml_tensor * b, ggml_tensor * x, int stride) {
        ggml_tensor * col = ggml_im2col(c, k, x, stride, 0, 1, 0, 1, 0, false, GGML_TYPE_F32);
        ggml_tensor * y   = ggml_mul_mat(c, ggml_reshape_2d(c, col, col->ne[0], col->ne[1] * col->ne[2]),
                                            ggml_reshape_2d(c, k, k->ne[0] * k->ne[1], k->ne[2]));
        return ggml_add(c, y, ggml_reshape_2d(c, b, 1, b->ne[0]));
    }

    void build() {
        ggml_init_params ip = { /*mem_size*/ ggml_tensor_overhead() * 1024 + ggml_graph_overhead(), nullptr, /*no_alloc*/ true };
        ctx = ggml_init(ip);
        ggml_context * c = ctx;
        const int hd = d / n_head;

        mel = ggml_new_tensor_2d(c, GGML_TYPE_F32, ST_FRAMES, ST_MEL);
        ggml_set_input(mel);

        ggml_tensor * x = ggml_gelu_erf(c, conv1d(c, get("enc.conv1.weight"), get("enc.conv1.bias"), mel, 1));
        x = ggml_gelu_erf(c, conv1d(c, get("enc.conv2.weight"), get("enc.conv2.bias"), x, 2)); // [400, 384]
        x = ggml_add(c, ggml_cont(c, ggml_transpose(c, x)), get("enc.pos_emb"));             // [384, 400]

        for (int l = 0; l < n_layer; l++) {
            const std::string p = "enc." + std::to_string(l) + ".";
            ggml_tensor * h = layer_norm(c, x, get(p + "attn_ln.weight"), get(p + "attn_ln.bias"));
            ggml_tensor * q = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, linear(c, h, p + "attn.q"), hd, n_head, n_ctx), 0, 2, 1, 3));
            ggml_tensor * k = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, linear(c, h, p + "attn.k", false), hd, n_head, n_ctx), 0, 2, 1, 3));
            ggml_tensor * v = ggml_cont(c, ggml_permute(c, ggml_reshape_3d(c, linear(c, h, p + "attn.v"), hd, n_head, n_ctx), 1, 2, 0, 3));
            ggml_tensor * kq = ggml_soft_max_ext(c, ggml_mul_mat(c, k, q), nullptr, 1.0f / std::sqrt((float) hd), 0.0f);
            ggml_tensor * o  = ggml_cont(c, ggml_permute(c, ggml_mul_mat(c, v, kq), 0, 2, 1, 3));
            x = ggml_add(c, x, linear(c, ggml_reshape_2d(c, o, d, n_ctx), p + "attn.o"));
            h = layer_norm(c, x, get(p + "ffn_ln.weight"), get(p + "ffn_ln.bias"));
            x = ggml_add(c, x, linear(c, ggml_gelu_erf(c, linear(c, h, p + "fc1")), p + "fc2"));
        }
        ggml_tensor * enc = layer_norm(c, x, get("enc.ln_post.weight"), get("enc.ln_post.bias")); // [384, 400]

        // attention pooling over time, then the classifier
        ggml_tensor * s = linear(c, ggml_tanh(c, linear(c, enc, "pool.0")), "pool.2");
        s = ggml_soft_max(c, ggml_reshape_2d(c, s, n_ctx, 1));
        ggml_tensor * pooled = ggml_mul_mat(c, ggml_cont(c, ggml_transpose(c, enc)), s);
        ggml_tensor * y = linear(c, pooled, "cls.0");
        y = ggml_gelu_erf(c, layer_norm(c, y, get("cls.ln.weight"), get("cls.ln.bias")));
        y = ggml_gelu_erf(c, linear(c, y, "cls.4"));
        prob = ggml_sigmoid(c, linear(c, y, "cls.6"));
        ggml_set_output(prob);

        gf = ggml_new_graph(c);
        ggml_build_forward_expand(gf, prob);
        galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            throw std::runtime_error("smart-turn: failed to allocate the compute buffer");
        }
    }

    // HF WhisperFeatureExtractor(chunk_length=8) with do_normalize=True
    void log_mel(std::vector<float> pcm, float * out) const {
        if ((int) pcm.size() > ST_SAMPLES) {
            pcm.erase(pcm.begin(), pcm.end() - ST_SAMPLES);
        }
        if ((int) pcm.size() < ST_SAMPLES) {
            pcm.insert(pcm.begin(), ST_SAMPLES - pcm.size(), 0.0f); // the padding goes in front
        }
        double mean = 0, var = 0;
        for (float v : pcm) {
            mean += v;
        }
        mean /= ST_SAMPLES;
        for (float v : pcm) {
            var += (v - mean) * (v - mean);
        }
        var /= ST_SAMPLES;
        const double inv = 1.0 / std::sqrt(var + 1e-7);
        for (float & v : pcm) {
            v = (float) ((v - mean) * inv);
        }

        const int pad = ST_FFT / 2, n_bins = ST_FFT / 2 + 1;
        std::vector<float> padded(ST_SAMPLES + 2 * pad);
        std::copy(pcm.begin(), pcm.end(), padded.begin() + pad);
        for (int i = 0; i < pad; i++) {
            padded[pad - 1 - i]              = pcm[i + 1];
            padded[pad + ST_SAMPLES + i]     = pcm[ST_SAMPLES - 2 - i];
        }
        std::vector<float> frame(ST_FFT), power(n_bins);
        std::vector<cf>    spec(ST_FFT);
        float mx = -1e30f;
        for (int f = 0; f < ST_FRAMES; f++) {
            for (int i = 0; i < ST_FFT; i++) {
                frame[i] = padded[f * ST_HOP + i] * hann[i];
            }
            fft_rec(tw.data(), frame.data(), 1, spec.data(), ST_FFT);
            for (int b = 0; b < n_bins; b++) {
                power[b] = std::norm(spec[b]);
            }
            for (int m = 0; m < ST_MEL; m++) {
                const float * fl = filters.data() + (size_t) m * n_bins;
                float acc = 0;
                for (int b = 0; b < n_bins; b++) {
                    acc += fl[b] * power[b];
                }
                const float v = std::log10(std::max(acc, 1e-10f));
                out[m * ST_FRAMES + f] = v;
                mx = std::max(mx, v);
            }
        }
        for (int i = 0; i < ST_MEL * ST_FRAMES; i++) {
            out[i] = (std::max(out[i], mx - 8.0f) + 4.0f) / 4.0f;
        }
    }

    float end_of_turn_prob(const float * pcm, size_t n) override {
        std::lock_guard<std::mutex> lock(mtx);
        std::vector<float> feats(ST_MEL * ST_FRAMES);
        log_mel(std::vector<float>(pcm, pcm + n), feats.data());
        ggml_backend_tensor_set(mel, feats.data(), 0, feats.size() * sizeof(float));
        ggml_cplan plan = ggml_graph_plan(gf, n_threads, nullptr);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.data();
        if (ggml_graph_compute(gf, &plan) != GGML_STATUS_SUCCESS) {
            return 0.0f;
        }
        float p = 0.0f;
        ggml_backend_tensor_get(prob, &p, 0, sizeof(p));
        return p;
    }
};

std::shared_ptr<server_turn_model> server_turn_model_load(const std::string & path, int n_threads) {
    auto m = std::make_shared<smart_turn_model>();
    gguf_init_params params = { /*no_alloc*/ false, /*ctx*/ &m->wctx };
    m->gctx = gguf_init_from_file(path.c_str(), params);
    if (!m->gctx) {
        LOG_ERR("%s: failed to load %s\n", __func__, path.c_str());
        return nullptr;
    }
    const int64_t arch = gguf_find_key(m->gctx, "general.architecture");
    if (arch < 0 || std::string(gguf_get_val_str(m->gctx, arch)) != "smart-turn") {
        LOG_ERR("%s: %s is not a smart-turn model\n", __func__, path.c_str());
        return nullptr;
    }
    auto u32 = [&](const char * key, int def) {
        const int64_t i = gguf_find_key(m->gctx, key);
        return i < 0 ? def : (int) gguf_get_val_u32(m->gctx, i);
    };
    m->n_layer   = u32("smart_turn.n_layer", 4);
    m->d         = u32("smart_turn.d_model", 384);
    m->n_head    = u32("smart_turn.n_head", 6);
    m->n_ctx     = u32("smart_turn.n_ctx", 400);
    m->n_threads = std::max(1, n_threads);
    try {
        ggml_tensor * fl = m->get("mel_filters"); // [201, 80]
        m->filters.assign((const float *) fl->data, (const float *) fl->data + ggml_nelements(fl));
        m->build();
    } catch (const std::exception & e) {
        LOG_ERR("%s: %s\n", __func__, e.what());
        return nullptr;
    }
    m->tw.resize(ST_FFT);
    m->hann.resize(ST_FFT);
    for (int i = 0; i < ST_FFT; i++) {
        m->tw[i]   = cf((float) std::cos(-2.0 * M_PI * i / ST_FFT), (float) std::sin(-2.0 * M_PI * i / ST_FFT));
        m->hann[i] = 0.5f - 0.5f * std::cos(2.0f * (float) M_PI * i / ST_FFT);
    }
    return m;
}
