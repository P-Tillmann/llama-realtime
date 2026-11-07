#include "server-asr.h"
#include "server-chat.h"

#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <future>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>

server_asr_params server_asr_params_parse(const std::string & spec) {
    server_asr_params p;
    std::stringstream ss(spec);
    std::string kv;
    while (std::getline(ss, kv, ',')) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) {
            if (!p.model.empty()) {
                throw std::invalid_argument("--asr-model: \"" + kv + "\" is not key=value");
            }
            p.model = kv;
            continue;
        }
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if      (k == "model")  { p.model  = v; }
        else if (k == "mmproj") { p.mmproj = v; }
        else if (k == "lang")   { p.lang   = v; }
        else {
            throw std::invalid_argument("--asr-model: unknown key \"" + k + "\" (model, mmproj, lang)");
        }
    }
    if (p.model.empty() || p.mmproj.empty()) {
        throw std::invalid_argument("--asr-model needs model=FILE and mmproj=FILE");
    }
    return p;
}

struct server_asr_turn {
    std::string         lang;
    server_asr::delta_fn on_delta;
    server_asr::done_fn  on_done;

    // under the mutex of the model
    std::vector<float> pending;   // audio not given to the worker yet
    bool               finishing = false;
    bool               cancelled = false;
    bool               done      = false;

    // worker thread only
    int                    seq = -1;
    mtmd::audio_stream_ptr stream;
    llama_tokens           prompt; // of the stream, it must outlive it
    common_sampler_ptr     smpl;
    llama_token            last   = LLAMA_TOKEN_NULL;
    llama_pos              n_past = 0;
    std::vector<float>     audio; // the whole turn, for models without an audio stream
    std::string            text;
    int32_t                n_in  = 0;
    int32_t                n_out = 0;
};

// the text of the transcription, without what some models put before it (Qwen3-ASR: "language English<asr_text>")
static std::string asr_clean(std::string s) {
    const size_t p = s.rfind("<asr_text>");
    if (p != std::string::npos) {
        s = s.substr(p + strlen("<asr_text>"));
    }
    const size_t a = s.find_first_not_of(" \t\n");
    const size_t b = s.find_last_not_of(" \t\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

class server_asr_model {
  public:
    server_asr_params p;
    std::string       name;
    llama_model *     model = nullptr;
    llama_context *   lctx  = nullptr;
    const llama_vocab * vocab = nullptr;
    mtmd::context_ptr mctx;
    common_chat_templates_ptr tmpls;
    llama_tokens      asr_prompt; // of the "asr" chat template, for streaming
    bool              stream_mode = false;
    int               rate        = 0;

    static constexpr int n_seq = 8; // turns transcribed at the same time

    server_asr_model(const server_asr_params & params, const common_params & base) : p(params) {
        const int n_cpu = base.cpuparams.n_threads > 0 ? base.cpuparams.n_threads : (int) std::max(1u, std::thread::hardware_concurrency());

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = base.n_gpu_layers;
        model = llama_model_load_from_file(p.model.c_str(), mp);
        if (!model) {
            throw std::runtime_error("failed to load the ASR model " + p.model);
        }
        vocab = llama_model_get_vocab(model);
        char buf[256] = {0};
        name = llama_model_meta_val_str(model, "general.name", buf, sizeof(buf)) > 0 ? std::string(buf) :
                                                                                        std::filesystem::path(p.model).stem().string();

        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 8192;
        cp.n_batch         = 2048;
        cp.n_ubatch        = 512;
        cp.n_seq_max       = n_seq;
        cp.kv_unified      = true;
        cp.n_threads       = n_cpu;
        cp.n_threads_batch = n_cpu;
        lctx = llama_init_from_model(model, cp);
        if (!lctx) {
            throw std::runtime_error("failed to create a context for the ASR model " + p.model);
        }
        mtmd_context_params mtp = mtmd_context_params_default();
        mtp.use_gpu       = base.mmproj_use_gpu;
        mtp.n_threads     = n_cpu;
        mtp.print_timings = false;
        mctx.reset(mtmd_init_from_file(p.mmproj.c_str(), model, mtp));
        if (!mctx) {
            throw std::runtime_error("failed to load the ASR mmproj " + p.mmproj);
        }
        if (!mtmd_support_audio(mctx.get())) {
            throw std::runtime_error(p.mmproj + " has no audio input");
        }
        rate  = mtmd_get_audio_sample_rate(mctx.get());
        tmpls = common_chat_templates_init(model, "");
        asr_prompt  = server_chat_get_asr_prompt(model);
        stream_mode = mtmd_support_audio_stream(mctx.get()) && !asr_prompt.empty();
        for (int i = n_seq - 1; i >= 0; i--) {
            free_seqs.push_back(i);
        }

        // the first transcription allocates the compute graphs: do it now, not in the first turn
        {
            server_asr_turn warm;
            warm.pending.assign(rate, 0.0f);
            step(warm, true);
            release(warm);
        }
        worker = std::thread([this]() { run(); });
        LOG_INF("%s: ASR \"%s\": %s, %d Hz, %s\n", __func__, name.c_str(), p.model.c_str(), rate,
                stream_mode ? "transcribes as the audio arrives" : "transcribes at the end of a turn");
    }

    ~server_asr_model() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stopping = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        turns.clear();
        mctx.reset();
        if (lctx) {
            llama_free(lctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }

    std::shared_ptr<server_asr_turn> open(const std::string & language, server_asr::delta_fn on_delta) {
        auto t = std::make_shared<server_asr_turn>();
        t->lang     = language.empty() ? p.lang : language;
        t->on_delta = std::move(on_delta);
        {
            std::lock_guard<std::mutex> lock(mtx);
            turns.push_back(t);
        }
        return t;
    }

    void push(const std::shared_ptr<server_asr_turn> & t, std::vector<float> pcm) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (t->finishing || t->cancelled) {
                return;
            }
            t->pending.insert(t->pending.end(), pcm.begin(), pcm.end());
        }
        cv.notify_one();
    }

    void finish(const std::shared_ptr<server_asr_turn> & t, server_asr::done_fn on_done) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            if (!t->finishing && !t->cancelled) {
                t->on_done   = std::move(on_done);
                t->finishing = true;
                on_done      = nullptr;
            }
        }
        if (on_done) {
            on_done(false, "", 0, 0);
            return;
        }
        cv.notify_one();
    }

    void cancel(const std::shared_ptr<server_asr_turn> & t) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            t->cancelled = true;
        }
        cv.notify_one();
    }

  private:
    std::thread                                   worker;
    std::mutex                                    mtx;
    std::condition_variable                       cv;
    std::vector<std::shared_ptr<server_asr_turn>> turns;
    std::vector<int>                              free_seqs; // worker thread only
    bool                                          stopping = false;

    // work to do for a turn, under the mutex
    static bool has_work(const server_asr_turn & t) {
        return !t.done && (t.cancelled || t.finishing || !t.pending.empty());
    }

    void run() {
        while (true) {
            std::vector<std::shared_ptr<server_asr_turn>> work;
            std::vector<bool>                             fin;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]() {
                    return stopping || std::any_of(turns.begin(), turns.end(), [](const auto & t) { return has_work(*t); });
                });
                if (stopping) {
                    return;
                }
                for (const auto & t : turns) {
                    if (!has_work(*t)) {
                        continue;
                    }
                    if (t->cancelled) {
                        t->done = true;
                    }
                    work.push_back(t);
                    fin.push_back(t->finishing);
                }
            }
            for (size_t i = 0; i < work.size(); i++) {
                auto & t = *work[i];
                bool   ok = true;
                if (!t.done) {
                    ok = step(t, fin[i]);
                }
                finish_turn(work[i], ok, fin[i]);
            }
        }
    }

    // a turn that has ended (its audio was complete at the step, cancelled or failed): its sequence is free, on_done
    // gets the text; a finish() that came during the step is taken at the next one, after the stream is flushed
    void finish_turn(const std::shared_ptr<server_asr_turn> & tp, bool ok, bool finished) {
        auto & t = *tp;
        bool cancelled;
        server_asr::done_fn on_done;
        {
            std::lock_guard<std::mutex> lock(mtx);
            cancelled = t.cancelled;
            if (!finished && !cancelled && ok) {
                return; // still open, more audio comes
            }
            t.done  = true;
            on_done = std::move(t.on_done);
            turns.erase(std::remove(turns.begin(), turns.end(), tp), turns.end());
        }
        release(t);
        if (on_done) {
            on_done(ok && !cancelled, asr_clean(t.text), t.n_in, t.n_out);
        }
    }

    void release(server_asr_turn & t) {
        if (t.seq >= 0) {
            llama_memory_seq_rm(llama_get_memory(lctx), t.seq, -1, -1);
            free_seqs.push_back(t.seq);
            t.seq = -1;
        }
        t.stream.reset();
        t.smpl.reset();
    }

    // takes the audio that has arrived, transcribes what it can; finishing: the audio of the turn is complete
    bool step(server_asr_turn & t, bool finishing) {
        std::vector<float> pcm;
        {
            std::lock_guard<std::mutex> lock(mtx);
            pcm.swap(t.pending);
        }
        if (t.seq < 0) {
            if (free_seqs.empty()) {
                LOG_ERR("%s: %s: too many turns at the same time\n", __func__, name.c_str());
                return false;
            }
            t.seq = free_seqs.back();
            free_seqs.pop_back();
            llama_memory_seq_rm(llama_get_memory(lctx), t.seq, -1, -1);
        }
        if (!stream_mode) {
            t.audio.insert(t.audio.end(), pcm.begin(), pcm.end());
            return finishing ? whole(t) : true;
        }
        return stream_step(t, pcm, finishing);
    }

    bool decode(server_asr_turn & t, const llama_token * tokens, const float * rows, int32_t n) {
        const int32_t n_embd = llama_model_n_embd_inp(model);
        llama_batch b = llama_batch_init(n, rows ? n_embd : 0, 1);
        for (int32_t i = 0; i < n; i++) {
            if (rows) {
                memcpy(b.embd + (size_t) i * n_embd, rows + (size_t) i * n_embd, n_embd * sizeof(float));
            } else {
                b.token[i] = tokens[i];
            }
            b.pos[i]       = t.n_past + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = t.seq;
            b.logits[i]    = i == n - 1;
        }
        b.n_tokens = n;
        const int ret = llama_decode(lctx, b);
        llama_batch_free(b);
        if (ret != 0) {
            LOG_ERR("%s: %s: decode failed\n", __func__, name.c_str());
            return false;
        }
        t.n_past += n;
        return true;
    }

    std::string piece(llama_token tok) const {
        return llama_vocab_is_control(vocab, tok) ? "" : common_token_to_piece(vocab, tok, false);
    }

    // the prompt of the "asr" template; a language goes in as the transcription prompt of the model names it
    // (LFM3: "Perform live ASR." -> "Perform live ASR in German.")
    llama_tokens stream_prompt(const std::string & lang) const {
        if (lang.empty()) {
            return asr_prompt;
        }
        const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls.get(), lang);
        const std::string base = common_chat_get_asr_prompt(tmpls.get(), "").system; // "Perform ASR."
        if (!preset.has_language || base.size() < 2 || preset.system.size() <= base.size()) {
            return asr_prompt;
        }
        // the words the language adds, e.g. " in German"
        const std::string add = preset.system.substr(base.size() - 1, preset.system.size() - base.size());
        std::string text = common_detokenize(vocab, asr_prompt, true);
        const std::string key = "ASR.";
        const size_t p = text.find(key);
        if (p == std::string::npos) {
            return asr_prompt;
        }
        text.insert(p + key.size() - 1, add);
        return common_tokenize(vocab, text, false, true);
    }

    // live ASR: the stream turns the audio into rows, one text token is sampled per step
    bool stream_step(server_asr_turn & t, const std::vector<float> & pcm, bool finishing) {
        if (!t.stream) {
            t.prompt    = stream_prompt(t.lang);
            auto sp = mtmd_audio_stream_params_default();
            sp.prompt   = t.prompt.data();
            sp.n_prompt = t.prompt.size();
            t.stream.reset(mtmd_audio_stream_init(mctx.get(), sp));
            if (!t.stream) {
                LOG_ERR("%s: %s: failed to open the audio stream\n", __func__, name.c_str());
                return false;
            }
            common_params_sampling sparams;
            sparams.temp = 0.0f;
            for (const auto & b : mtmd_helper::audio_stream_logit_bias(mctx.get(), model)) {
                sparams.logit_bias.push_back(b);
            }
            t.smpl.reset(common_sampler_init(model, sparams));
            t.last   = LLAMA_TOKEN_NULL;
            t.n_past = 0;
        }
        if (!pcm.empty() && mtmd_audio_stream_push(t.stream.get(), pcm.data(), pcm.size()) != 0) {
            return false;
        }
        if (finishing && mtmd_audio_stream_flush(t.stream.get(), true) != 0) {
            return false;
        }
        while (true) {
            const int32_t n = mtmd_audio_stream_step(t.stream.get(), t.last);
            if (n == -2) {
                LOG_ERR("%s: %s: audio stream error\n", __func__, name.c_str());
                return false;
            }
            if (n == 0) {
                return true; // more audio is needed
            }
            if (n < 0) {
                return true; // the end of the stream
            }
            if (mtmd_audio_stream_is_seq_start(t.stream.get())) {
                // the model restarts its context (long turns), the text goes on
                llama_memory_seq_rm(llama_get_memory(lctx), t.seq, -1, -1);
                t.n_past = 0;
                if (!t.text.empty() && t.text.back() != ' ') {
                    t.text += ' ';
                }
            }
            const llama_token * tokens = mtmd_audio_stream_get_output_tokens(t.stream.get());
            const float *       rows   = tokens ? nullptr : mtmd_audio_stream_get_output_embd(t.stream.get());
            if (!decode(t, tokens, rows, n)) {
                return false;
            }
            t.n_in += n;
            t.last = common_sampler_sample(t.smpl.get(), lctx, -1);
            common_sampler_accept(t.smpl.get(), t.last, true);
            t.n_out++;
            const std::string s = piece(t.last);
            if (!s.empty()) {
                t.text += s;
                if (t.on_delta) {
                    t.on_delta(s);
                }
            }
        }
    }

    // a turn at once: the transcription prompt of the chat template with the audio, then greedy decoding
    bool whole(server_asr_turn & t) {
        if (t.audio.empty()) {
            return true;
        }
        const common_chat_prompt_preset preset = common_chat_get_asr_prompt(tmpls.get(), t.lang);
        std::string user = preset.user;
        if (!t.lang.empty() && !preset.has_language) {
            user += " (language: " + t.lang + ")";
        }
        user += mtmd_default_marker();
        common_chat_templates_inputs inputs;
        if (!preset.system.empty()) {
            common_chat_msg m;
            m.role    = "system";
            m.content = preset.system;
            inputs.messages.push_back(m);
        }
        common_chat_msg m;
        m.role    = "user";
        m.content = user;
        inputs.messages.push_back(m);
        inputs.add_generation_prompt = true;
        inputs.enable_thinking       = false;
        std::string prompt;
        try {
            prompt = common_chat_templates_apply(tmpls.get(), inputs).prompt.str();
        } catch (const std::exception & e) {
            LOG_ERR("%s: %s: chat template: %s\n", __func__, name.c_str(), e.what());
            return false;
        }

        mtmd::bitmap bmp(mtmd_bitmap_init_from_audio(t.audio.size(), t.audio.data()));
        mtmd::input_chunks chunks(mtmd_input_chunks_init());
        mtmd_input_text inp = { prompt.c_str(), prompt.size(), true, true };
        const mtmd_bitmap * bmps[] = { bmp.ptr.get() };
        if (!bmp.ptr || mtmd_tokenize(mctx.get(), chunks.ptr.get(), &inp, bmps, 1) != 0) {
            LOG_ERR("%s: %s: failed to tokenize the audio\n", __func__, name.c_str());
            return false;
        }
        t.n_past = 0;
        llama_pos new_n_past = 0;
        if (mtmd_helper_eval_chunks(mctx.get(), lctx, chunks.ptr.get(), 0, t.seq, llama_n_batch(lctx), true, &new_n_past) != 0) {
            LOG_ERR("%s: %s: failed to encode the audio\n", __func__, name.c_str());
            return false;
        }
        t.n_past = new_n_past;
        t.n_in   = (int32_t) mtmd_helper_get_n_tokens(chunks.ptr.get());

        // about 8 tokens per second of speech are plenty
        const int32_t n_max = 64 + (int32_t) (t.audio.size() / std::max(1, rate)) * 8;
        common_params_sampling sparams;
        sparams.temp = 0.0f;
        common_sampler_ptr smpl(common_sampler_init(model, sparams));
        std::string out;
        for (int32_t i = 0; i < n_max; i++) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                if (t.cancelled) {
                    return false;
                }
            }
            const llama_token tok = common_sampler_sample(smpl.get(), lctx, -1);
            common_sampler_accept(smpl.get(), tok, true);
            if (llama_vocab_is_eog(vocab, tok)) {
                break;
            }
            t.n_out++;
            out += piece(tok);
            if (!decode(t, &tok, nullptr, 1)) {
                return false;
            }
        }
        t.text = out;
        return true;
    }
};

server_asr::server_asr(const server_asr_params & params, const common_params & base)
    : impl(std::make_unique<server_asr_model>(params, base)) {}

server_asr::~server_asr() = default;

const std::string & server_asr::name() const { return impl->name; }
int  server_asr::sample_rate() const { return impl->rate; }
bool server_asr::streaming() const { return impl->stream_mode; }

std::shared_ptr<server_asr_turn> server_asr::open(const std::string & language, delta_fn on_delta) {
    return impl->open(language, std::move(on_delta));
}

void server_asr::push(const std::shared_ptr<server_asr_turn> & turn, std::vector<float> pcm) {
    impl->push(turn, std::move(pcm));
}

void server_asr::finish(const std::shared_ptr<server_asr_turn> & turn, done_fn on_done) {
    impl->finish(turn, std::move(on_done));
}

void server_asr::cancel(const std::shared_ptr<server_asr_turn> & turn) {
    impl->cancel(turn);
}

bool server_asr::transcribe(const std::vector<float> & pcm, const std::string & language, std::string & text,
                            int32_t & n_input, int32_t & n_output) {
    std::promise<bool> done;
    auto turn = open(language);
    push(turn, pcm);
    finish(turn, [&](bool ok, const std::string & t, int32_t n_in, int32_t n_out) {
        text     = t;
        n_input  = n_in;
        n_output = n_out;
        done.set_value(ok);
    });
    return done.get_future().get();
}
