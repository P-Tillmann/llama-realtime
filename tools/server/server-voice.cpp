#include "server-voice.h"

#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "sampling.h"

#include <algorithm>
#include <cstring>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

server_voice_params server_voice_params_parse(const std::string & spec) {
    server_voice_params p;
    std::stringstream ss(spec);
    std::string kv;
    while (std::getline(ss, kv, ',')) {
        const size_t eq = kv.find('=');
        if (eq == std::string::npos) {
            // a bare value is the model file
            if (!p.model.empty()) {
                throw std::invalid_argument("--voice-model: \"" + kv + "\" is not key=value");
            }
            p.model = kv;
            continue;
        }
        const std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
        if      (k == "model")   { p.model   = v; }
        else if (k == "mmproj")  { p.mmproj  = v; }
        else if (k == "name")    { p.name    = v; }
        else if (k == "lang")    { p.lang    = v; }
        else if (k == "voice")   { p.voice   = v; }
        else if (k == "speaker") { p.speaker = v; }
        else {
            throw std::invalid_argument("--voice-model: unknown key \"" + k + "\" (model, mmproj, name, lang, voice, speaker)");
        }
    }
    if (p.model.empty() || p.mmproj.empty()) {
        throw std::invalid_argument("--voice-model needs model=FILE and mmproj=FILE");
    }
    if (p.name.empty()) {
        p.name = std::filesystem::path(p.model).stem().string();
    }
    return p;
}

//
// one voice model: its own llama context, the jobs run on its worker thread
//

class server_voice_model {
  public:
    server_voice_params p;
    llama_model *       model = nullptr;
    llama_context *     lctx  = nullptr;
    mtmd::context_ptr   mctx;
    common_chat_templates_ptr tmpls;
    mtmd_helper::gen_audio    gen;
    int                       rate = 0;
    std::vector<std::string>  voices; // predefined voices of the model

    server_voice_model(const server_voice_params & params, const common_params & base) : p(params) {
        // the language model writes while the voice speaks: the voice gets half of the CPU threads
        const int n_cpu     = base.cpuparams.n_threads > 0 ? base.cpuparams.n_threads : (int) std::max(1u, std::thread::hardware_concurrency());
        const int n_threads = std::max(1, n_cpu / 2);

        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = base.n_gpu_layers;
        model = llama_model_load_from_file(p.model.c_str(), mp);
        if (!model) {
            throw std::runtime_error("failed to load the voice model " + p.model);
        }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 4096;
        cp.n_batch         = 2048;
        cp.n_ubatch        = 512;
        cp.n_seq_max       = 2; // the second sequence is the unconditioned one of classifier-free guidance (Chatterbox)
        cp.kv_unified      = true;
        cp.embeddings      = true; // the hidden states drive the audio stages
        cp.n_threads       = n_threads;
        cp.n_threads_batch = n_threads;
        lctx = llama_init_from_model(model, cp);
        if (!lctx) {
            throw std::runtime_error("failed to create a context for the voice model " + p.model);
        }
        mtmd_context_params mtp = mtmd_context_params_default();
        mtp.use_gpu   = base.mmproj_use_gpu;
        mtp.n_threads = n_threads;
        mtp.print_timings = false;
        mctx.reset(mtmd_init_from_file(p.mmproj.c_str(), model, mtp));
        if (!mctx) {
            throw std::runtime_error("failed to load the voice mmproj " + p.mmproj);
        }
        const auto info = mtmd_gen_audio_get_info(mctx.get());
        if (info.type == MTMD_GEN_AUDIO_TYPE_NONE) {
            throw std::runtime_error(p.mmproj + " has no audio generation");
        }
        rate = info.sample_rate;
        for (int32_t i = 0; const char * v = mtmd_gen_audio_voice_name(mctx.get(), i); i++) {
            voices.push_back(v);
        }
        if (!voices.empty()) {
            if (p.voice.empty()) {
                p.voice = voices[0];
            } else if (std::find(voices.begin(), voices.end(), p.voice) == voices.end()) {
                throw std::runtime_error("voice model " + p.name + " has no voice \"" + p.voice + "\"");
            }
            if (!p.speaker.empty()) {
                throw std::runtime_error("voice model " + p.name + " has predefined voices, speaker= cannot be used");
            }
        }
        tmpls = common_chat_templates_init(model, "");
        gen.init(lctx, mctx.get());
        if (!gen.valid()) {
            throw std::runtime_error("failed to initialize audio generation for " + p.name);
        }
        // the first synthesis allocates the compute graphs: do it now, not in the first response
        {
            server_voice_job warm;
            warm.text = "Hello.";
            int32_t n = 0;
            speak(warm, n);
        }
        worker = std::thread([this]() { run(); });
        LOG_INF("%s: voice \"%s\": %s, %d Hz, %zu predefined voices%s\n", __func__, p.name.c_str(), p.model.c_str(), rate,
                voices.size(), p.voice.empty() ? "" : (", default " + p.voice).c_str());
    }

    ~server_voice_model() {
        {
            std::lock_guard<std::mutex> lock(mtx);
            stopping = true;
        }
        cv.notify_all();
        if (worker.joinable()) {
            worker.join();
        }
        gen = mtmd_helper::gen_audio();
        mctx.reset();
        if (lctx) {
            llama_free(lctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }

    void submit(server_voice_job job) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            queue.push_back(std::move(job));
        }
        cv.notify_one();
    }

  private:
    std::thread                  worker;
    std::mutex                   mtx;
    std::condition_variable      cv;
    std::deque<server_voice_job> queue;
    bool                         stopping = false;

    void run() {
        while (true) {
            server_voice_job job;
            {
                std::unique_lock<std::mutex> lock(mtx);
                cv.wait(lock, [this]() { return stopping || !queue.empty(); });
                if (stopping) {
                    return;
                }
                job = std::move(queue.front());
                queue.pop_front();
            }
            int32_t n_frames = 0;
            const bool ok = speak(job, n_frames);
            if (job.on_done) {
                job.on_done(ok, n_frames);
            }
        }
    }

    bool is_cancelled(const server_voice_job & job) const {
        return job.cancelled && job.cancelled->load();
    }

    bool emit(const server_voice_job & job) {
        int32_t      sr   = 0;
        const char * data = nullptr;
        size_t       len  = 0;
        if (gen.get_output(&sr, &data, &len) != 0) {
            return false;
        }
        if (len > 0 && job.on_audio && !is_cancelled(job)) {
            job.on_audio((const float *) data, len / sizeof(float));
        }
        return true;
    }

    bool speak(const server_voice_job & job, int32_t & n_frames) {
        if (is_cancelled(job) || job.text.empty()) {
            return !is_cancelled(job);
        }
        const std::string voice = job.voice.empty() ? p.voice : job.voice;

        // a model with a "tts" chat template (e.g. LFM3-Audio) speaks the prompt rendered from it
        std::string prompt = common_chat_get_tts_prompt(tmpls.get(), job.text, p.lang, voice);
        if (prompt.empty()) {
            prompt = job.text;
        }
        mtmd_helper::gen_audio::inp inp;
        inp.set_prompt(prompt);
        inp.set_lang(p.lang);
        inp.data.seq_id   = 0;
        inp.data.stream   = true;
        inp.data.out_type = MTMD_HELPER_GEN_AUDIO_OUTTYPE_PCM;
        if (!voices.empty()) {
            inp.set_speaker_ref(mtmd::bitmap_ptr(mtmd_gen_audio_get_voice(mctx.get(), voice.c_str())));
        } else if (!p.speaker.empty()) {
            auto w = mtmd_helper_bitmap_init_from_file(mctx.get(), p.speaker.c_str(), false, mtmd_helper_init_opt_default());
            inp.set_speaker_ref(mtmd::bitmap_ptr(w.bitmap));
        }

        common_params_sampling sp;
        sp.penalty_repeat = 1.05f; // as /tts: without it the backbone can repeat the utterance
        sp.penalty_last_n = 512;
        sp.seed           = std::random_device{}();
        inp.data.top_k = sp.top_k;
        inp.data.top_p = sp.top_p;
        inp.data.seed  = sp.seed;

        // every job starts from an empty context (both sequences)
        llama_memory_clear(llama_get_memory(lctx), true); // seq_rm cannot empty recurrent memory (e.g. LFM3-Audio)
        if (gen.set_input(inp.get()) != 0) {
            LOG_ERR("%s: %s: failed to set the input\n", __func__, p.name.c_str());
            return false;
        }
        for (int32_t ret; (ret = gen.step_prompt(llama_n_batch(lctx))) != 0; ) {
            if (ret < 0) {
                LOG_ERR("%s: %s: prompt failed\n", __func__, p.name.c_str());
                return false;
            }
        }
        common_sampler_ptr smpl(common_sampler_init(model, sp));
        if (!smpl) {
            return false;
        }
        auto sample = [&]() {
            const llama_token t = common_sampler_sample(smpl.get(), lctx, -1);
            common_sampler_accept(smpl.get(), t, true);
            return t;
        };
        llama_token sampled = sample();
        const float * h0 = llama_get_embeddings_ith(lctx, -1);
        std::vector<float> h_state(h0, h0 + llama_model_n_embd(model));

        const int32_t max_frames = 4096;
        bool stop = false;
        while (!stop && n_frames < max_frames && !is_cancelled(job)) {
            const float * h_next = nullptr;
            if (gen.step_gen(sampled, h_state.data(), &h_next, &stop) != 0) {
                LOG_ERR("%s: %s: generation failed\n", __func__, p.name.c_str());
                return false;
            }
            if (!h_next) {
                break; // end of speech without a new frame
            }
            n_frames++;
            h_state.assign(h_next, h_next + llama_model_n_embd(model));
            if (!emit(job)) {
                return false;
            }
            if (!stop) {
                sampled = sample();
            }
        }
        if (gen.flush() != 0 || !emit(job)) {
            return false;
        }
        return !is_cancelled(job);
    }
};

server_voices::server_voices(const std::vector<server_voice_params> & params, const common_params & base) {
    for (const auto & p : params) {
        for (const auto & m : voice_models) {
            if (m->p.name == p.name) {
                throw std::runtime_error("two voice models are named \"" + p.name + "\", give one a name=");
            }
        }
        voice_models.push_back(std::make_unique<server_voice_model>(p, base));
        const auto & m = *voice_models.back();
        info inf;
        inf.name        = m.p.name;
        inf.lang        = m.p.lang;
        inf.sample_rate = m.rate;
        inf.voices      = m.voices;
        if (!m.p.voice.empty()) {
            // the default voice first
            inf.voices.erase(std::remove(inf.voices.begin(), inf.voices.end(), m.p.voice), inf.voices.end());
            inf.voices.insert(inf.voices.begin(), m.p.voice);
        }
        infos.push_back(std::move(inf));
    }
}

server_voices::~server_voices() = default;

int server_voices::find(const std::string & selector, std::string & voice) const {
    voice.clear();
    if (infos.empty()) {
        return -1;
    }
    if (selector.empty()) {
        return 0;
    }
    const size_t slash = selector.find('/');
    const std::string name = slash == std::string::npos ? selector : selector.substr(0, slash);
    const std::string v    = slash == std::string::npos ? "" : selector.substr(slash + 1);
    for (size_t i = 0; i < infos.size(); i++) {
        if (infos[i].name == name) {
            voice = v;
            return (int) i;
        }
    }
    if (slash != std::string::npos) {
        return -1;
    }
    // a voice name: the first model that has it
    for (size_t i = 0; i < infos.size(); i++) {
        if (std::find(infos[i].voices.begin(), infos[i].voices.end(), selector) != infos[i].voices.end()) {
            voice = selector;
            return (int) i;
        }
    }
    return -1;
}

void server_voices::submit(int model, server_voice_job job) {
    voice_models.at(model)->submit(std::move(job));
}

//
// segmenter
//

// characters that only format the text, a voice would read them out
static std::string speakable(std::string s) {
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return c == '*' || c == '#' || c == '`' || c == '_'; }), s.end());
    const size_t a = s.find_first_not_of(" \t\n");
    const size_t b = s.find_last_not_of(" \t\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

std::vector<server_voice_segmenter::piece> server_voice_segmenter::push(const std::string & text) {
    buf += text;
    std::vector<piece> out;
    // a piece ends at a sentence end, a line break, or (the first one, to start speaking early) at a comma
    // after enough words; the end must be followed by a space so that "3.5" or "e.g." in the middle do not cut
    size_t start = 0;
    for (size_t i = 0; i + 1 < buf.size(); i++) {
        const char c = buf[i];
        const bool space_after = buf[i + 1] == ' ' || buf[i + 1] == '\n';
        bool cut = (c == '\n') || ((c == '.' || c == '!' || c == '?' || c == ';' || c == ':') && space_after);
        if (!cut && first && c == ',' && space_after && i - start >= 24) {
            cut = true;
        }
        // the first piece may also end before a conjunction, a natural pause, to start speaking early
        if (!cut && first && c == ' ' && i - start >= 24) {
            static const char * conj[] = {"and ", "but ", "or ", "because ", "so ", "und ", "aber ", "oder ", "weil ", "denn ",
                                          "sondern ", "et ", "mais ", "y ", "pero ", "e ", "ma "};
            for (const char * w : conj) {
                if (buf.compare(i + 1, strlen(w), w) == 0) {
                    cut = true;
                    break;
                }
            }
        }
        if (!cut) {
            continue;
        }
        const std::string t = speakable(buf.substr(start, i + 1 - start));
        if (t.size() >= 2) {
            out.push_back({t, i + 1 - start});
            first = false;
        } else if (!out.empty()) {
            out.back().raw_len += i + 1 - start;
        } else {
            continue; // too short to speak alone: it goes with the next piece
        }
        start = i + 1;
    }
    buf.erase(0, start);
    return out;
}

server_voice_segmenter::piece server_voice_segmenter::flush() {
    piece p = {speakable(buf), buf.size()};
    buf.clear();
    first = true;
    return p;
}
