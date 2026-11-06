#include "server-turn.h"

#include <algorithm>
#include <cmath>

namespace {

struct vad_energy : server_vad {
    int   rate;
    float noise_db = -60.0f; // noise floor, follows the level between speech

    explicit vad_energy(int rate) : rate(rate) {}

    int sample_rate() const override { return rate; }
    int frame_size() const override { return rate * 32 / 1000; }

    float speech_prob(const float * frame) override {
        const int n = frame_size();
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += (double) frame[i] * frame[i];
        }
        const float db = 10.0f * std::log10((float) (sum / n) + 1e-12f);
        if (db < -50.0f) {
            noise_db = std::max(-90.0f, std::min(noise_db, db));
            return 0.0f;
        }
        const float p = 1.0f / (1.0f + std::exp(-(db - noise_db - 12.0f) / 2.0f));
        if (p < 0.5f) {
            // the floor drops fast and rises slowly, so speech does not pull it up
            noise_db += (db < noise_db ? 0.3f : 0.02f) * (db - noise_db);
            noise_db  = std::max(-90.0f, std::min(-20.0f, noise_db));
        }
        return p;
    }

    void reset() override { noise_db = -60.0f; }
};

// semantic_vad: how long a silence ends the turn without a turn model, the longest silence with one,
// and the end-of-turn probability that ends it earlier
struct eagerness_cfg {
    int   fallback_ms;
    int   max_ms;
    float threshold;
};

eagerness_cfg eagerness_of(const std::string & e) {
    if (e == "low") {
        return {1500, 8000, 0.7f};
    }
    if (e == "high") {
        return {500, 2000, 0.3f};
    }
    return {800, 4000, 0.5f}; // medium, auto
}

constexpr int MIN_SPEECH_MS     = 64;  // speech needed before it counts as the start of a turn
constexpr int TURN_CHECK_MS     = 200; // semantic_vad: silence between two turn model checks
constexpr int TURN_WINDOW_S     = 8;   // audio of the turn the turn model sees
constexpr float END_HYSTERESIS  = 0.15f;

} // namespace

std::unique_ptr<server_vad> server_vad_energy(int sample_rate) {
    return std::make_unique<vad_energy>(sample_rate);
}

server_turn_detector::server_turn_detector(const server_turn_params & params, std::unique_ptr<server_vad> vad,
                                           server_turn_model * turn_model)
    : prm(params), vad(std::move(vad)), turn_model(turn_model) {
    rate    = this->vad->sample_rate();
    frame_n = this->vad->frame_size();
}

void server_turn_detector::response_done() {
    idle_n = 0;
}

std::vector<server_turn_action> server_turn_detector::push(const float * pcm, size_t n) {
    std::vector<server_turn_action> out;
    pending.insert(pending.end(), pcm, pcm + n);
    size_t i = 0;
    for (; i + frame_n <= pending.size(); i += frame_n) {
        frame(pending.data() + i, out);
    }
    pending.erase(pending.begin(), pending.begin() + i);
    return out;
}

static void add_audio(std::vector<server_turn_action> & out, const float * x, size_t n) {
    if (n == 0) {
        return;
    }
    if (out.empty() || out.back().kind != server_turn_action::AUDIO) {
        out.push_back({server_turn_action::AUDIO, 0, {}});
    }
    out.back().audio.insert(out.back().audio.end(), x, x + n);
}

void server_turn_detector::frame(const float * f, std::vector<server_turn_action> & out) {
    const float p = vad->speech_prob(f);
    n_samples += frame_n;

    if (!speaking) {
        if (p >= prm.threshold) {
            candidate.insert(candidate.end(), f, f + frame_n);
            speech_n += frame_n;
            if (speech_n * 1000 >= (int64_t) MIN_SPEECH_MS * rate) {
                const int64_t start = n_samples - speech_n - (int64_t) prefix.size();
                out.push_back({server_turn_action::SPEECH_STARTED, std::max<int64_t>(0, start) * 1000 / rate, {}});
                const std::vector<float> head(prefix.begin(), prefix.end());
                add_audio(out, head.data(), head.size());
                add_audio(out, candidate.data(), candidate.size());
                turn = head;
                turn.insert(turn.end(), candidate.begin(), candidate.end());
                prefix.clear();
                candidate.clear();
                speaking     = true;
                silence_n    = 0;
                last_check_n = 0;
                idle_n       = 0;
            }
            return;
        }
        // not (yet) speech: the candidate frames fall back into the padding
        prefix.insert(prefix.end(), candidate.begin(), candidate.end());
        prefix.insert(prefix.end(), f, f + frame_n);
        candidate.clear();
        speech_n = 0;
        const size_t keep = (size_t) prm.prefix_padding_ms * rate / 1000;
        while (prefix.size() > keep) {
            prefix.pop_front();
        }
        idle_n += frame_n;
        if (prm.idle_timeout_ms > 0 && idle_n * 1000 >= (int64_t) prm.idle_timeout_ms * rate) {
            out.push_back({server_turn_action::IDLE_TIMEOUT, n_samples * 1000 / rate, {}});
            idle_n = INT64_MIN / 2; // once, until the next turn or response
        }
        return;
    }

    add_audio(out, f, frame_n);
    turn.insert(turn.end(), f, f + frame_n);
    const size_t window = (size_t) TURN_WINDOW_S * rate;
    if (turn.size() > window) {
        turn.erase(turn.begin(), turn.end() - window);
    }
    if (p < prm.threshold - END_HYSTERESIS) {
        silence_n += frame_n;
    } else if (p >= prm.threshold) {
        silence_n    = 0;
        last_check_n = 0;
    }
    if (silence_n > 0 && turn_ended()) {
        out.push_back({server_turn_action::SPEECH_STOPPED, n_samples * 1000 / rate, {}});
        speaking  = false;
        silence_n = 0;
        idle_n    = 0;
        turn.clear();
    }
}

bool server_turn_detector::turn_ended() {
    const int64_t silence_ms = silence_n * 1000 / rate;
    if (prm.type != "semantic_vad") {
        return silence_ms >= prm.silence_duration_ms;
    }
    const eagerness_cfg cfg = eagerness_of(prm.eagerness);
    if (!turn_model) {
        return silence_ms >= cfg.fallback_ms;
    }
    if (silence_ms >= cfg.max_ms) {
        return true;
    }
    if (silence_ms < TURN_CHECK_MS || (silence_n - last_check_n) * 1000 < (int64_t) TURN_CHECK_MS * rate) {
        return false;
    }
    last_check_n = silence_n;
    return turn_model->end_of_turn_prob(turn.data(), turn.size()) >= cfg.threshold;
}
